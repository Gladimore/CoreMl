#!/usr/bin/env python3
"""
convert_to_coreml.py — turn a training checkpoint into what AIPlayer.dylib loads.

    pip install torch==2.7.0 coremltools
    python convert_to_coreml.py --checkpoint checkpoints/best.pt --out_dir build

The checkpoint kind is detected from its weights:

  temporal (train_temporal.py / TemporalSwipeNet) -> one model
      SwipeTemporal.mlmodel frames (1,K,H,W) gray at tick offsets e.g. [0,1,2,4,8]
                            (index 0 = newest), raw 0..255 -> det_logit (1,), dir_logits (1,4)
      The device keeps a ring of the last max(offset)+1 frames on the training clock.

  simple  (train_simple.py / SimpleSwipeAnnotator) -> one model
      SwipeSimple.mlmodel   frame (1,1,H,W) gray, raw 0..255 -> det_logit (1,), dir_logits (1,4)

  causal  (train_causal.py / CausalSwipeAnnotator) -> two models
      SwipeEncoder.mlmodel  frame (1,2,H,W) (gray, diff), raw 0..255 -> feat (1,128)
      SwipeHead.mlmodel     fast_feats (1,8,128) [t-7..t], slow_feats (1,2,128) [t-19,t-13]
                            -> det_logit (1,), dir_logits (1,4)
      The device caches one encoder feature per tick and re-runs the head from a
      zero GRU state each tick, which is exactly what training evaluated.

It always writes AIPlayerConfig.json next to the models. AIPlayer reads the model
kind, input size, tick rate and gate thresholds from that file, so the dylib never
has to be rebuilt to change them, and a model can't be paired with the wrong
pipeline.

Every run checks that the exported computation equals model.forward() in PyTorch,
and on macOS that the saved Core ML models agree too; any mismatch aborts.

Why "neuralnetwork" and hand-unrolled GRUs: Core ML's compiler rejects the
while_loop/gather ops coremltools emits for nn.GRU (while still exiting 0), and
mlprogram AOT compilation was seen emitting an empty .mlmodelc. Every op used here
has been supported since Core ML 1.0.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

TORCH_ATOL = 1e-4       # exported torch graph vs. model.forward()
COREML_ATOL = 2e-2      # Core ML (CPU) vs. model.forward()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def detect_kind(state: dict) -> str:
    if any(k.startswith("fast_gru.") for k in state):
        return "causal"
    if "squeeze.0.weight" in state:
        return "temporal"
    if "trunk.0.weight" in state:
        return "simple"
    sys.exit("[error] unrecognised checkpoint: expected Simple/Temporal/CausalSwipeAnnotator weights")


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state" not in ckpt:
        sys.exit(f"[error] {path} has no 'model_state'; keys: {list(ckpt.keys())}")
    state = ckpt["model_state"]
    kind = detect_kind(state)
    arch = ckpt.get("arch") or {}
    args = ckpt.get("args") or {}
    if not isinstance(args, dict):
        args = vars(args)

    if kind == "simple":
        from model_simple import build_simple_model
        model = build_simple_model(arch)
    elif kind == "temporal":
        from model_temporal import build_temporal_model
        model = build_temporal_model(arch)
    else:
        from model_causal import build_causal_model
        model = build_causal_model(arch)
    model.load_state_dict(state)            # strict: any shape/key mismatch is fatal
    model.eval()
    return ckpt, kind, model, args


# ─────────────────────────────────────────────────────────────────────────────
# Traceable wrappers
# ─────────────────────────────────────────────────────────────────────────────
# Same maths as each model's forward() minus the channels_last hint, which is a
# memory-layout detail with no numeric effect that can trip tracing. Dropout is a
# no-op in eval() and is left out.

class SimpleWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frame):
        feat = self.model.trunk(self.model.cnn(frame * (1.0 / 255.0)))
        return self.model.det_head(feat).squeeze(-1), self.model.dir_head(feat)


class EncoderWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frame):
        return self.model.cnn(frame * (1.0 / 255.0))


def gru_step(x, h, gru: nn.GRU):
    """One timestep through every layer of `gru` (PyTorch's documented GRU equations)."""
    layer_in, states = x, []
    for layer in range(h.shape[0]):
        w_ih, w_hh = getattr(gru, f"weight_ih_l{layer}"), getattr(gru, f"weight_hh_l{layer}")
        b_ih, b_hh = getattr(gru, f"bias_ih_l{layer}"), getattr(gru, f"bias_hh_l{layer}")
        h_prev = h[layer]
        i_r, i_z, i_n = (layer_in @ w_ih.t() + b_ih).split(h_prev.shape[-1], dim=-1)
        h_r, h_z, h_n = (h_prev @ w_hh.t() + b_hh).split(h_prev.shape[-1], dim=-1)
        r, z = torch.sigmoid(i_r + h_r), torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h_new = (1 - z) * n + z * h_prev
        states.append(h_new)
        layer_in = h_new
    return torch.stack(states, dim=0)


class HeadWrapper(nn.Module):
    def __init__(self, model, fast_len: int, slow_len: int):
        super().__init__()
        self.model, self.fast_len, self.slow_len = model, fast_len, slow_len
        self.register_buffer("h0_fast", torch.zeros(model.fast_layers, 1, model.hidden), persistent=False)
        self.register_buffer("h0_slow", torch.zeros(model.slow_layers, 1, model.slow_hidden), persistent=False)

    def forward(self, fast_feats, slow_feats):
        h_fast, h_slow = self.h0_fast, self.h0_slow
        for t in range(self.fast_len):
            h_fast = gru_step(fast_feats[:, t], h_fast, self.model.fast_gru)
        for t in range(self.slow_len):
            h_slow = gru_step(slow_feats[:, t], h_slow, self.model.slow_gru)
        fused = torch.cat([h_fast[-1], h_slow[-1]], dim=-1)
        return self.model.det_head(fused).squeeze(-1), self.model.dir_head(fused)


# ─────────────────────────────────────────────────────────────────────────────
# Export plans: what each kind produces and how to check it
# ─────────────────────────────────────────────────────────────────────────────

def _max_diff(a, b) -> float:
    return float(np.abs(np.asarray(a, np.float32).reshape(-1) - np.asarray(b, np.float32).reshape(-1)).max())


def _check(label: str, det, ref_det, direction, ref_dir, atol: float):
    d1, d2 = _max_diff(det, ref_det), _max_diff(direction, ref_dir)
    print(f"[verify] {label}: det diff {d1:.2e}, dir diff {d2:.2e}")
    if max(d1, d2) > atol:
        sys.exit(f"[error] {label} does not match model.forward() — aborting")


def plan_simple(model, img: int):
    wrapper = SimpleWrapper(model).eval()
    models = [dict(name="SwipeSimple", module=wrapper, example=(torch.zeros(1, 1, img, img),),
                   inputs=[("frame", (1, 1, img, img))], outputs=["det_logit", "dir_logits"],
                   description="SimpleSwipeAnnotator: one grayscale frame -> swipe logits",
                   input_docs={"frame": "grayscale frame, RAW 0..255 (normalised inside the model)"})]

    def samples():
        g = torch.Generator().manual_seed(0)
        return [torch.randint(0, 256, (1, 1, img, img), generator=g).float() for _ in range(3)]

    def verify_torch():
        for i, x in enumerate(samples()):
            with torch.no_grad():
                ref = model(x)
                got = wrapper(x)
            _check(f"torch parity #{i}", got[0], ref[0], got[1], ref[1], TORCH_ATOL)

    def verify_coreml(paths):
        import coremltools as ct
        ml = ct.models.MLModel(paths["SwipeSimple"], compute_units=ct.ComputeUnit.CPU_ONLY)
        x = samples()[0]
        with torch.no_grad():
            ref = model(x)
        out = ml.predict({"frame": x.numpy().astype(np.float32)})
        _check("Core ML parity", out["det_logit"], ref[0].numpy(), out["dir_logits"], ref[1].numpy(), COREML_ATOL)

    return models, verify_torch, verify_coreml, {}


class TemporalWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, frames):
        m = self.model
        h = m.trunk(m.squeeze(m.features(frames * (1.0 / 255.0))))
        return m.det_head(h).squeeze(-1), m.dir_head(h)


def plan_temporal(model, img: int):
    k = len(model.offsets)
    wrapper = TemporalWrapper(model).eval()
    models = [dict(name="SwipeTemporal", module=wrapper, example=(torch.zeros(1, k, img, img),),
                   inputs=[("frames", (1, k, img, img))], outputs=["det_logit", "dir_logits"],
                   description="TemporalSwipeNet: stacked grayscale frames -> swipe logits",
                   input_docs={"frames": f"grayscale frames at tick offsets {list(model.offsets)} "
                                         "(index 0 = newest), RAW 0..255"})]

    def samples():
        g = torch.Generator().manual_seed(0)
        return [torch.randint(0, 256, (1, k, img, img), generator=g).float() for _ in range(3)]

    def verify_torch():
        for i, x in enumerate(samples()):
            with torch.no_grad():
                ref, got = model(x), wrapper(x)
            _check(f"torch parity #{i}", got[0], ref[0], got[1], ref[1], TORCH_ATOL)

    def verify_coreml(paths):
        import coremltools as ct
        ml = ct.models.MLModel(paths["SwipeTemporal"], compute_units=ct.ComputeUnit.CPU_ONLY)
        x = samples()[0]
        with torch.no_grad():
            ref = model(x)
        out = ml.predict({"frames": x.numpy().astype(np.float32)})
        _check("Core ML parity", out["det_logit"], ref[0].numpy(), out["dir_logits"], ref[1].numpy(), COREML_ATOL)

    return models, verify_torch, verify_coreml, {"frame_offsets": list(model.offsets)}


def plan_causal(model, img: int):
    from model_causal import FAST_LEN, FEAT_DIM, SLOW_OFFSETS
    slow_len = len(SLOW_OFFSETS)
    enc = EncoderWrapper(model).eval()
    head = HeadWrapper(model, FAST_LEN, slow_len).eval()
    models = [
        dict(name="SwipeEncoder", module=enc, example=(torch.zeros(1, 2, img, img),),
             inputs=[("frame", (1, 2, img, img))], outputs=["feat"],
             description="CausalSwipeAnnotator per-frame CNN encoder",
             input_docs={"frame": "(gray, diff), RAW 0..255 (normalised inside the model)"}),
        dict(name="SwipeHead", module=head,
             example=(torch.zeros(1, FAST_LEN, FEAT_DIM), torch.zeros(1, slow_len, FEAT_DIM)),
             inputs=[("fast_feats", (1, FAST_LEN, FEAT_DIM)), ("slow_feats", (1, slow_len, FEAT_DIM))],
             outputs=["det_logit", "dir_logits"],
             description="CausalSwipeAnnotator windowed head (zero-state GRUs)",
             input_docs={"fast_feats": "features of ticks [t-7 .. t], oldest first",
                         "slow_feats": "features of ticks [t-19, t-13], oldest first"}),
    ]

    def window(seed):
        g = torch.Generator().manual_seed(seed)
        return (torch.randint(0, 256, (1, FAST_LEN, 2, img, img), generator=g).float(),
                torch.randint(0, 256, (1, slow_len, 2, img, img), generator=g).float())

    def verify_torch():
        for seed in range(3):
            fast, slow = window(seed)
            with torch.no_grad():
                ref = model(fast, slow)
                ff = torch.stack([enc(fast[:, i]) for i in range(FAST_LEN)], dim=1)
                sf = torch.stack([enc(slow[:, i]) for i in range(slow_len)], dim=1)
                got = head(ff, sf)
            _check(f"torch parity #{seed}", got[0], ref[0], got[1], ref[1], TORCH_ATOL)

    def verify_coreml(paths):
        import coremltools as ct
        e = ct.models.MLModel(paths["SwipeEncoder"], compute_units=ct.ComputeUnit.CPU_ONLY)
        h = ct.models.MLModel(paths["SwipeHead"], compute_units=ct.ComputeUnit.CPU_ONLY)

        def feat(x):
            out = e.predict({"frame": x.numpy().astype(np.float32)})
            return np.asarray(out["feat"], np.float32).reshape(1, FEAT_DIM)

        fast, slow = window(123)
        with torch.no_grad():
            ref = model(fast, slow)
        ff = np.stack([feat(fast[:, i]) for i in range(FAST_LEN)], axis=1)
        sf = np.stack([feat(slow[:, i]) for i in range(slow_len)], axis=1)
        out = h.predict({"fast_feats": ff, "slow_feats": sf})
        _check("Core ML parity", out["det_logit"], ref[0].numpy(), out["dir_logits"], ref[1].numpy(), COREML_ATOL)

    extra = {"fast_len": FAST_LEN, "slow_offsets": list(SLOW_OFFSETS), "feat_dim": FEAT_DIM}
    return models, verify_torch, verify_coreml, extra


# ─────────────────────────────────────────────────────────────────────────────
# Conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert(spec: dict):
    import coremltools as ct
    with torch.no_grad():
        traced = torch.jit.trace(spec["module"], spec["example"], check_trace=True)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name=n, shape=s, dtype=np.float32) for n, s in spec["inputs"]],
        outputs=[ct.TensorType(name=n) for n in spec["outputs"]],
        minimum_deployment_target=ct.target.iOS14,
        convert_to="neuralnetwork",
        compute_units=ct.ComputeUnit.ALL,
    )
    ml.short_description = spec["description"]
    for name, doc in spec["input_docs"].items():
        ml.input_description[name] = doc
    return ml


def pick_threshold(ckpt: dict, args: dict, override):
    if override is not None:
        return float(override), "--fire_threshold"
    if ckpt.get("best_threshold") is not None:
        return float(ckpt["best_threshold"]), "checkpoint best_threshold (validation sweep)"
    if args.get("det_threshold") is not None:
        return float(args["det_threshold"]), "checkpoint args.det_threshold"
    return 0.5, "default"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out_dir", default="build")
    ap.add_argument("--img_size", type=int, default=None, help="default: from checkpoint, else 128")
    ap.add_argument("--tick_hz", type=float, default=None,
                    help="how often the device evaluates a frame. causal: must equal the training "
                         "video frame rate (default: checkpoint, else 24). simple: no temporal "
                         "dependence, so higher just means lower latency (default 30)")
    ap.add_argument("--fire_threshold", type=float, default=None,
                    help="detection probability that fires a swipe (default: from checkpoint)")
    ap.add_argument("--skip_verify", action="store_true", help="skip the Core ML runtime check (macOS)")
    args = ap.parse_args()

    print(f"[1/4] Loading {args.checkpoint}")
    ckpt, kind, model, ck_args = load_checkpoint(args.checkpoint)
    img = args.img_size or int(ck_args.get("img_size", 128))
    tick_hz = args.tick_hz or (30.0 if kind == "simple" else float(ck_args.get("fps", 24.0)))
    print(f"[info] kind={kind} arch={ckpt.get('arch')} img={img} epoch={ckpt.get('epoch')} "
          f"val_f1={ckpt.get('val_f1')}")

    plan = {"simple": plan_simple, "temporal": plan_temporal, "causal": plan_causal}[kind]
    specs, verify_torch, verify_coreml, extra = plan(model, img)

    print("[2/4] Verifying the exported computation against model.forward()")
    verify_torch()

    print("[3/4] Converting (neuralnetwork)")
    os.makedirs(args.out_dir, exist_ok=True)
    paths = {}
    for spec in specs:
        path = os.path.join(args.out_dir, spec["name"] + ".mlmodel")
        convert(spec).save(path)
        paths[spec["name"]] = path
        print(f"[info] wrote {path}")

    if args.skip_verify or sys.platform != "darwin":
        print("[verify] Core ML runtime check skipped" + ("" if args.skip_verify else " (needs macOS)"))
    else:
        verify_coreml(paths)

    fire, source = pick_threshold(ckpt, ck_args, args.fire_threshold)
    config = {
        "model_type": kind,
        "models": [s["name"] + ".mlmodelc" for s in specs],
        "img_size": img,
        "tick_hz": tick_hz,
        # Gate: fire once when prob >= fire_threshold, then wait for prob < release_threshold,
        # except that a different move with dir confidence >= rearm_dir_confidence may fire
        # mid-burst (0 disables). tools/diagnose.py --dataset_dir calibrates fire_threshold.
        "fire_threshold": round(fire, 4),
        "release_threshold": round(fire * 0.5, 4),
        "min_dir_confidence": 0.0,
        "rearm_dir_confidence": 0.6,
        "cooldown_s": 0.15,
        # Linear correction applied to the 0..255 gray input before the model
        # (gray' = gray * input_gain + input_bias). tools/diagnose.py suggests values.
        "input_gain": 1.0,
        "input_bias": 0.0,
        # >0: save this many preprocessed frames + predictions to Documents/AIPlayerDump.
        "debug_dump_frames": 0,
        **extra,
        "source": {"checkpoint": os.path.basename(args.checkpoint), "epoch": ckpt.get("epoch"),
                   "val_f1": ckpt.get("val_f1"), "fire_threshold_from": source},
    }
    cfg_path = os.path.join(args.out_dir, "AIPlayerConfig.json")
    with open(cfg_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[4/4] wrote {cfg_path} (fire_threshold={fire:.3f} from {source}, tick_hz={tick_hz:g})")
    print("Next: xcrun coremlcompiler compile <each .mlmodel> <out_dir>; ship the .mlmodelc folders "
          "and AIPlayerConfig.json next to AIPlayer.dylib.")


if __name__ == "__main__":
    main()
