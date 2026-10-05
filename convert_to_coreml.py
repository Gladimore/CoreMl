#!/usr/bin/env python3
"""
convert_to_coreml.py — export a CausalSwipeAnnotator checkpoint as two Core ML
models that reproduce the windowed `forward()` used in training exactly.

    pip install torch==2.7.0 coremltools
    python convert_to_coreml.py --checkpoint checkpoints/best.pt --out_dir . --img_size 128

Outputs (classic "neuralnetwork" spec, single .mlmodel files):

    SwipeEncoder.mlmodel
        in : frame      (1, 2, H, W) float32   (gray, diff) pair, RAW 0..255 values
        out: feat       (1, 128)     float32
      Run once per new frame; the host keeps the last 20 features.

    SwipeHead.mlmodel
        in : fast_feats (1, 8, 128)  float32   features of frames [t-7 ... t], oldest first
             slow_feats (1, 2, 128)  float32   features of frames [t-19, t-13], oldest first
        out: det_logit  (1,)         sigmoid() -> detection probability
             dir_logits (1, 4)       softmax() -> [UP, DOWN, LEFT, RIGHT]
      Both GRUs are unrolled from a zero hidden state, exactly like training.

Why two models and no carried hidden state
------------------------------------------
Training evaluates every sample from a zero hidden state over exactly 8 fast
and 2 slow frames. A streaming export that carries GRU state across the whole
session computes a different function that the checkpoint was never validated
on. Caching per-frame features on the host and re-running the (cheap) temporal
head per tick reproduces training exactly while still running the CNN once per
frame.

Why GRUs are unrolled by hand
-----------------------------
coremltools lowers nn.GRU to a while_loop with gather/scatter, which Core ML's
compiler rejects ("Unsupported opset for gather op") while still exiting 0.
`manual_gru_step` implements PyTorch's documented GRU equations with the
trained weights; tracing fully unrolls the fixed-length Python loops into
straight-line ops. Equivalence is asserted on every run.

Why "neuralnetwork" instead of "mlprogram"
------------------------------------------
mlprogram's AOT compilation was observed to emit an empty .mlmodelc silently.
Every op used here (conv, batchnorm, relu, pool, linear, elementwise, concat,
sigmoid, tanh) has been supported since Core ML 1.0. The format caps the
deployment target at iOS 14, which is a floor, not a ceiling.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from model_causal import (CausalSwipeAnnotator, FAST_LEN, FEAT_DIM, SLOW_OFFSETS,
                          build_causal_model)

SLOW_LEN = len(SLOW_OFFSETS)
TORCH_ATOL = 1e-4          # unrolled torch pipeline vs. model.forward()
COREML_ATOL = 2e-2         # Core ML (CPU) vs. model.forward()


# ─────────────────────────────────────────────────────────────────────────────
# Hand-unrolled GRU
# ─────────────────────────────────────────────────────────────────────────────

def manual_gru_step(x: torch.Tensor, h: torch.Tensor, gru: nn.GRU) -> torch.Tensor:
    """
    One timestep through every layer of `gru`, using its trained weights.

        r  = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
        z  = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
        n  = tanh   (W_in x + b_in + r * (W_hn h + b_hn))
        h' = (1 - z) * n + z * h

    x: (B, input)   h: (layers, B, hidden)   returns h': (layers, B, hidden)
    """
    layer_input = x
    new_states = []
    for layer in range(h.shape[0]):
        w_ih = getattr(gru, f"weight_ih_l{layer}")
        w_hh = getattr(gru, f"weight_hh_l{layer}")
        b_ih = getattr(gru, f"bias_ih_l{layer}")
        b_hh = getattr(gru, f"bias_hh_l{layer}")
        h_prev = h[layer]
        hidden = h_prev.shape[-1]

        i_r, i_z, i_n = (layer_input @ w_ih.t() + b_ih).split(hidden, dim=-1)
        h_r, h_z, h_n = (h_prev @ w_hh.t() + b_hh).split(hidden, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h_new = (1 - z) * n + z * h_prev

        new_states.append(h_new)
        layer_input = h_new
    return torch.stack(new_states, dim=0)


def unrolled_gru(feats: torch.Tensor, h0: torch.Tensor, gru: nn.GRU, steps: int) -> torch.Tensor:
    """feats (B, steps, in), zero h0 (layers, B, hidden) -> last-step top-layer output (B, hidden)."""
    h = h0
    for t in range(steps):
        h = manual_gru_step(feats[:, t], h, gru)
    return h[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Traceable wrappers
# ─────────────────────────────────────────────────────────────────────────────

class EncoderWrapper(nn.Module):
    def __init__(self, model: CausalSwipeAnnotator):
        super().__init__()
        self.model = model

    def forward(self, frame):
        # Same maths as model._cnn_features minus the channels_last hint, which is a
        # memory-layout detail with no numeric effect and can trip tracing/conversion.
        return self.model.cnn(frame * (1.0 / 255.0))     # (1, 128)


class HeadWrapper(nn.Module):
    def __init__(self, model: CausalSwipeAnnotator):
        super().__init__()
        self.model = model
        self.register_buffer("h0_fast", torch.zeros(model.fast_layers, 1, model.hidden),
                             persistent=False)
        self.register_buffer("h0_slow", torch.zeros(model.slow_layers, 1, model.slow_hidden),
                             persistent=False)

    def forward(self, fast_feats, slow_feats):
        h_fast = unrolled_gru(fast_feats, self.h0_fast, self.model.fast_gru, FAST_LEN)
        h_slow = unrolled_gru(slow_feats, self.h0_slow, self.model.slow_gru, SLOW_LEN)
        fused = torch.cat([h_fast, h_slow], dim=-1)      # dropout is a no-op in eval()
        return self.model.det_head(fused).squeeze(-1), self.model.dir_head(fused)


# ─────────────────────────────────────────────────────────────────────────────
# Verification
# ─────────────────────────────────────────────────────────────────────────────

def random_window(img: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    fast = torch.randint(0, 256, (1, FAST_LEN, 2, img, img), generator=g).float()
    slow = torch.randint(0, 256, (1, SLOW_LEN, 2, img, img), generator=g).float()
    return fast, slow


def verify_torch_parity(model, enc: EncoderWrapper, head: HeadWrapper, img: int):
    """Encoder -> cached features -> unrolled head must equal model.forward()."""
    for seed in range(3):
        fast, slow = random_window(img, seed)
        with torch.no_grad():
            ref_det, ref_dir = model(fast, slow)
            f_feats = torch.stack([enc(fast[:, i]) for i in range(FAST_LEN)], dim=1)
            s_feats = torch.stack([enc(slow[:, i]) for i in range(SLOW_LEN)], dim=1)
            det, direction = head(f_feats, s_feats)
        d1 = (det - ref_det).abs().max().item()
        d2 = (direction - ref_dir).abs().max().item()
        print(f"[verify] torch parity seed={seed}: det diff {d1:.2e}, dir diff {d2:.2e}")
        if max(d1, d2) > TORCH_ATOL:
            sys.exit("[error] exported computation does not match model.forward() — aborting")


def verify_coreml(enc_path: str, head_path: str, model, img: int):
    """Run the saved Core ML models (CPU, fp32) and compare to model.forward(). macOS only."""
    if sys.platform != "darwin":
        print("[verify] Core ML runtime check skipped (needs macOS)")
        return
    import coremltools as ct

    enc = ct.models.MLModel(enc_path, compute_units=ct.ComputeUnit.CPU_ONLY)
    head = ct.models.MLModel(head_path, compute_units=ct.ComputeUnit.CPU_ONLY)

    def feat(frame_t):
        out = enc.predict({"frame": frame_t.numpy().astype(np.float32)})
        return np.asarray(out["feat"], dtype=np.float32).reshape(1, FEAT_DIM)

    fast, slow = random_window(img, seed=123)
    with torch.no_grad():
        ref_det, ref_dir = model(fast, slow)

    f_feats = np.stack([feat(fast[:, i]) for i in range(FAST_LEN)], axis=1)   # (1, 8, 128)
    s_feats = np.stack([feat(slow[:, i]) for i in range(SLOW_LEN)], axis=1)   # (1, 2, 128)
    out = head.predict({"fast_feats": f_feats, "slow_feats": s_feats})
    det = np.asarray(out["det_logit"], dtype=np.float32).reshape(-1)
    direction = np.asarray(out["dir_logits"], dtype=np.float32).reshape(1, 4)

    d1 = float(np.abs(det - ref_det.numpy().reshape(-1)).max())
    d2 = float(np.abs(direction - ref_dir.numpy()).max())
    print(f"[verify] Core ML parity: det diff {d1:.2e}, dir diff {d2:.2e}")
    if max(d1, d2) > COREML_ATOL:
        sys.exit("[error] Core ML output does not match model.forward() — aborting")


# ─────────────────────────────────────────────────────────────────────────────
# Conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert(module: nn.Module, example: tuple, inputs: list, outputs: list):
    import coremltools as ct

    with torch.no_grad():
        traced = torch.jit.trace(module, example, check_trace=True)
    return ct.convert(
        traced,
        inputs=[ct.TensorType(name=n, shape=tuple(s), dtype=np.float32) for n, s in inputs],
        outputs=[ct.TensorType(name=n) for n in outputs],
        minimum_deployment_target=ct.target.iOS14,
        convert_to="neuralnetwork",
        compute_units=ct.ComputeUnit.ALL,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to a train_causal.py checkpoint")
    ap.add_argument("--out_dir", default=".", help="Directory for SwipeEncoder/SwipeHead .mlmodel")
    ap.add_argument("--img_size", type=int, default=None,
                    help="Frame H=W (default: checkpoint args, else 128)")
    ap.add_argument("--skip_verify", action="store_true", help="Skip the Core ML runtime check")
    args = ap.parse_args()

    print(f"[1/5] Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state" not in ckpt or "arch" not in ckpt:
        sys.exit(f"[error] not a train_causal.py checkpoint; keys: {list(ckpt.keys())}")

    ckpt_args = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        ckpt_args = vars(ckpt_args)
    img = args.img_size or int(ckpt_args.get("img_size", 128))
    print(f"[info] arch={ckpt['arch']} img_size={img} epoch={ckpt.get('epoch')} "
          f"val_f1={ckpt.get('val_f1')}")

    print("[2/5] Building model")
    model = build_causal_model(ckpt["arch"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    enc_w, head_w = EncoderWrapper(model).eval(), HeadWrapper(model).eval()

    print("[3/5] Verifying unrolled pipeline against model.forward()")
    verify_torch_parity(model, enc_w, head_w, img)

    print("[4/5] Converting with coremltools (neuralnetwork)")
    enc_ml = convert(enc_w, (torch.zeros(1, 2, img, img),),
                     inputs=[("frame", (1, 2, img, img))], outputs=["feat"])
    head_ml = convert(head_w, (torch.zeros(1, FAST_LEN, FEAT_DIM), torch.zeros(1, SLOW_LEN, FEAT_DIM)),
                      inputs=[("fast_feats", (1, FAST_LEN, FEAT_DIM)),
                              ("slow_feats", (1, SLOW_LEN, FEAT_DIM))],
                      outputs=["det_logit", "dir_logits"])

    for label, ml in (("SwipeEncoder", enc_ml), ("SwipeHead", head_ml)):
        spec = ml.get_spec()
        ins = ", ".join(f"{i.name}:{i.type.multiArrayType.dataType}" for i in spec.description.input)
        print(f"[info] {label} declared input dtypes (enum): {ins}")

    enc_ml.short_description = "CausalSwipeAnnotator per-frame CNN encoder"
    enc_ml.input_description["frame"] = "(gray, diff) pair, RAW 0-255 values (normalised inside the model)"
    enc_ml.output_description["feat"] = "128-d frame feature"
    head_ml.short_description = "CausalSwipeAnnotator windowed temporal head (zero-state GRUs)"
    head_ml.input_description["fast_feats"] = "features of frames [t-7 .. t], oldest first"
    head_ml.input_description["slow_feats"] = "features of frames [t-19, t-13], oldest first"
    head_ml.output_description["det_logit"] = "sigmoid() -> detection probability at t"
    head_ml.output_description["dir_logits"] = "softmax() -> [UP, DOWN, LEFT, RIGHT]"

    os.makedirs(args.out_dir, exist_ok=True)
    enc_path = os.path.join(args.out_dir, "SwipeEncoder.mlmodel")
    head_path = os.path.join(args.out_dir, "SwipeHead.mlmodel")
    print(f"[5/5] Saving {enc_path} and {head_path}")
    enc_ml.save(enc_path)
    head_ml.save(head_path)

    if not args.skip_verify:
        verify_coreml(enc_path, head_path, model, img)

    print("\nNext: xcrun coremlcompiler compile SwipeEncoder.mlmodel . && "
          "xcrun coremlcompiler compile SwipeHead.mlmodel .\n"
          "Ship both .mlmodelc folders next to AIPlayer.dylib.")


if __name__ == "__main__":
    main()
