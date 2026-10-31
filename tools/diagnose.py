#!/usr/bin/env python3
"""
diagnose.py — find out WHY the on-device model misbehaves, from a device dump.

1. Set "debug_dump_frames": 600 in AIPlayerConfig.json, inject, play ~30 s with
   AI ON, then copy Documents/AIPlayerDump/<time>/ off the phone.
2. python tools/diagnose.py --dump <that folder> --checkpoint checkpoints/best.pt \
       --dataset_dir <training dataset dir with frames.bin + meta.npz>

It answers three questions:

  A. Does Core ML compute what PyTorch computes on the SAME frames?
     (No -> conversion / fp16 problem; try "compute_units": "cpu_only".)
  B. Do device frames look like training frames? Intensity percentiles, a per-row
     profile (status bar, overlay, letterboxing) and a side-by-side PNG.
     (No -> preprocessing/domain shift. If it's a global brightness/contrast
     difference it prints input_gain / input_bias for AIPlayerConfig.json.)
  C. Is the model more trigger-happy on device frames than on training frames?
     (Mean detection probability and the share of frames above fire_threshold.)
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from convert_to_coreml import load_checkpoint  # noqa: E402


def load_dump(path: str):
    with open(os.path.join(path, "config.json")) as f:
        cfg = json.load(f)
    S = int(cfg.get("img_size", 128))
    b64_path = os.path.join(path, "frames.b64")
    if os.path.exists(b64_path):                      # current format: one base64 frame per line
        import base64
        with open(b64_path) as f:
            frames = [np.frombuffer(base64.b64decode(line.strip()), np.uint8)
                      for line in f if line.strip()]
        frames = np.stack([fr for fr in frames if fr.size == S * S]).reshape(-1, S, S)
    else:                                             # older builds: raw binary
        raw = np.fromfile(os.path.join(path, "frames.u8"), dtype=np.uint8)
        if raw.size >= 30 and (raw[:30].reshape(-1, 3) == [0xEF, 0xBF, 0xBD]).all(axis=1).mean() > 0.5:
            sys.exit("[error] frames.u8 was corrupted in transfer: something treated it as text and "
                     "replaced every byte >= 128 with U+FFFD. Use a build that writes frames.b64, or copy "
                     "the file with a binary-safe method (Finder/iTunes file sharing, AirDrop of the folder).")
        frames = raw[: (raw.size // (S * S)) * S * S].reshape(-1, S, S)
    rows = []
    with open(os.path.join(path, "ticks.csv")) as f:
        next(f)
        for line in f:
            p = line.strip().split(",")
            if len(p) < 9:
                continue
            rows.append(dict(tick=int(p[0]), held=p[1] == "1", has_pred=p[2] == "1",
                             det=float(p[3]) if p[3] else np.nan,
                             dirs=[float(x) if x else np.nan for x in p[4:8]], fired=p[8] == "1"))
    n = min(len(frames), len(rows))
    return cfg, frames[:n], rows[:n]


def diff_channel(cur: np.ndarray, prev: np.ndarray) -> np.ndarray:
    d = (cur.astype(np.int32) - prev.astype(np.int32)) * 127
    return np.clip(np.floor_divide(d, 255) + 127, 0, 255).astype(np.uint8)


@torch.no_grad()
def torch_probs(kind, model, frames, held, cfg, want):
    """det prob for each index in `want`, recomputed in PyTorch from the dumped frames."""
    if kind == "simple":
        out = []
        for i in range(0, len(want), 256):
            x = torch.from_numpy(frames[want[i:i + 256]][:, None].astype(np.float32))
            out.append(torch.sigmoid(model(x)[0]))
        return torch.cat(out).numpy() if out else np.zeros(0)

    # causal: rebuild (gray, diff) per tick exactly like the device, then windows
    fast_len = int(cfg.get("fast_len", 8))
    slow = [int(o) for o in cfg.get("slow_offsets", [19, 13])]
    diffs = np.full_like(frames, 127)
    for t in range(1, len(frames)):
        if not held[t]:
            diffs[t] = diff_channel(frames[t], frames[t - 1])
    feats = []
    for i in range(0, len(frames), 256):
        x = torch.from_numpy(np.stack([frames[i:i + 256], diffs[i:i + 256]], 1).astype(np.float32))
        feats.append(model._cnn_features(x))
    feats = torch.cat(feats)
    res = []
    for t in want:
        ff = feats[t - fast_len + 1: t + 1][None]
        sf = feats[[t - o for o in slow]][None]
        res.append(torch.sigmoid(model.heads_from_features(ff, sf)[0]).item())
    return np.array(res)


@torch.no_grad()
def training_probs(kind, model, train, cfg, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "simple":
        idx = np.sort(rng.choice(len(train), size=min(n, len(train)), replace=False))
        return torch_probs(kind, model, np.asarray(train[idx]), np.zeros(len(idx), bool), cfg, list(range(len(idx))))
    span = max(int(cfg.get("fast_len", 8)), max(cfg.get("slow_offsets", [19, 13])) + 1)
    out = []
    for t in rng.integers(span, len(train), size=min(n // 4, 400)):
        clip = np.asarray(train[t - span: t + 1])
        out.append(torch_probs(kind, model, clip, np.zeros(len(clip), bool), cfg, [len(clip) - 1])[0])
    return np.array(out)


def fit_gain_bias(device: np.ndarray, train: np.ndarray):
    q = np.arange(2, 99)
    a, b = np.percentile(device, q), np.percentile(train, q)
    gain, bias = np.polyfit(a, b, 1)
    return float(gain), float(bias), float(np.abs(a - b).mean()), float(np.abs(a * gain + bias - b).mean())


def save_side_by_side(device, train, path, n=6):
    try:
        from PIL import Image
    except ImportError:
        print("  (pip install pillow for the side-by-side image)")
        return
    di = np.linspace(0, len(device) - 1, n).astype(int)
    ti = np.random.default_rng(1).choice(len(train), n, replace=False)
    top = np.concatenate([device[i] for i in di], 1)
    bot = np.concatenate([np.asarray(train[i]) for i in ti], 1)
    sep = np.full((4, top.shape[1]), 255, np.uint8)
    img = np.concatenate([top, sep, bot], 0)
    Image.fromarray(img).resize((img.shape[1] * 3, img.shape[0] * 3), Image.NEAREST).save(path)
    print(f"  wrote {path}  (top: device, bottom: training)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset_dir", default=None, help="training dataset (frames.bin + meta.npz)")
    ap.add_argument("--out", default=None, help="where to write the PNG (default: the dump folder)")
    a = ap.parse_args()

    cfg, frames, rows = load_dump(a.dump)
    ckpt, kind, model, _ = load_checkpoint(a.checkpoint)
    if kind != cfg.get("model_type"):
        sys.exit(f"[error] dump was made with a {cfg.get('model_type')} model but the checkpoint is {kind}")
    held = np.array([r["held"] for r in rows])
    fps = float(cfg.get("tick_hz", 24))
    fire = float(cfg.get("fire_threshold", 0.5))
    print(f"dump: {len(frames)} ticks ({len(frames) / fps:.1f}s at {fps:g} Hz), {held.sum()} held, "
          f"{sum(r['fired'] for r in rows)} fires; model={kind}; fire_threshold={fire}")

    # ── A: Core ML vs PyTorch on identical inputs ────────────────────────────
    print("\nA. Core ML vs PyTorch on the same frames")
    want = [i for i, r in enumerate(rows) if r["has_pred"]]
    if want:
        dev = np.array([rows[i]["det"] for i in want])
        ref = torch_probs(kind, model, frames, held, cfg, want)
        err = np.abs(dev - ref)
        flips = int(((dev >= fire) != (ref >= fire)).sum())
        print(f"  det prob |device - torch|: mean {err.mean():.4f}  max {err.max():.4f}; "
              f"fire decision differs on {flips}/{len(want)} ticks")
        if err.max() > 0.05 or flips:
            print("  -> Core ML output disagrees with PyTorch. Try \"compute_units\": \"cpu_only\" in the config;"
                  " if that fixes it, it's fp16 precision on GPU/ANE.")
        else:
            print("  -> OK: the converted model is faithful. Any problem is in what it is shown.")
    else:
        print("  no predictions in the dump (causal model needs a warm-up; dump more ticks)")

    # ── B: device frames vs training frames ──────────────────────────────────
    train = None
    if a.dataset_dir:
        meta = np.load(os.path.join(a.dataset_dir, "meta.npz"))
        T, S = int(meta["T_total"]), int(meta["img_size"])
        train = np.memmap(os.path.join(a.dataset_dir, "frames.bin"), dtype="uint8", mode="r", shape=(T, S, S))
        if S != frames.shape[1]:
            sys.exit(f"[error] dataset img_size {S} != device {frames.shape[1]}")

    def pct(x):
        return " ".join(f"{v:5.1f}" for v in np.percentile(x, [1, 5, 25, 50, 75, 95, 99]))

    print("\nB. Do device frames look like training frames?")
    print(f"  percentiles 1/5/25/50/75/95/99  device:   {pct(frames)}")
    if train is not None:
        sample = np.asarray(train[np.sort(np.random.default_rng(0).choice(len(train), min(3000, len(train)),
                                                                          replace=False))])
        print(f"                                   training: {pct(sample)}")
        gain, bias, before, after = fit_gain_bias(frames, sample)
        print(f"  quantile fit device->training: gain {gain:.3f}, bias {bias:+.1f} "
              f"(mean percentile gap {before:.1f} -> {after:.1f} levels)")
        if before > 4 and after < before * 0.6:
            g0, b0 = float(cfg.get("input_gain", 1)), float(cfg.get("input_bias", 0))
            print(f"  -> global intensity shift. Set in AIPlayerConfig.json: "
                  f"\"input_gain\": {g0 * gain:.3f}, \"input_bias\": {b0 * gain + bias:.1f}")

        rd = np.abs(frames.mean(axis=(0, 2)) - sample.mean(axis=(0, 2)))
        worst = np.argsort(rd)[::-1][:5]
        print(f"  row-mean gap: median {np.median(rd):.1f}; worst rows {sorted(worst.tolist())} "
              f"({rd[worst].min():.1f}-{rd[worst].max():.1f} levels)")
        if rd[worst].max() > 25:
            print("  -> some rows differ a lot (status bar, overlay or letterboxing?). Check the PNG.")
        save_side_by_side(frames, sample, os.path.join(a.out or a.dump, "device_vs_training.png"))

    # ── C: is the model more trigger-happy on device frames? ─────────────────
    print("\nC. Detection behaviour")
    dets = np.array([r["det"] for r in rows if r["has_pred"]])
    if len(dets):
        print(f"  device:   mean det prob {dets.mean():.3f}, above fire_threshold {np.mean(dets >= fire):.1%}, "
              f"fires/s {sum(r['fired'] for r in rows) / (len(rows) / fps):.2f}")
    if train is not None:
        tp = training_probs(kind, model, train, cfg)
        print(f"  training: mean det prob {tp.mean():.3f}, above fire_threshold {np.mean(tp >= fire):.1%}")
        if len(dets) and dets.mean() > 2 * tp.mean() + 0.02:
            print("  -> the model fires far more on device frames than on training frames: a domain shift"
                  " (section B), not bad luck. A model trained with label-dependent augmentation"
                  " (the old loader) is especially sensitive to this.")


if __name__ == "__main__":
    main()
