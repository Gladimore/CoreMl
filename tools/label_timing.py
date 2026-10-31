#!/usr/bin/env python3
"""
label_timing.py — when does the game visibly react to a labelled swipe?

    python tools/label_timing.py --dataset_dir dataset [--fps 24]

For every labelled swipe at frame t it measures, frame by frame from t-12 to
t+12, how far the picture shifts in the swipe's direction (LEFT/RIGHT: horizontal
shift, UP/DOWN: vertical shift), and compares that with random non-swipe frames.
The first offset where the shift is clearly above chance is the reaction onset R.

If R <= 0, the labelled frame already shows the move. A single-frame model
trained on those labels learns to recognise swipes that already happened, which
is exactly "the AI copies my moves and never moves on its own". The script
prints the --label_lead / --react_delay to train with.
"""

import argparse
import math
import os

import numpy as np
import pandas as pd

OFFSETS = np.arange(-12, 13)
MAX_SHIFT = 4


def best_shift(a: np.ndarray, b: np.ndarray, axis: int) -> int:
    """Integer shift s in [-MAX_SHIFT, MAX_SHIFT] along `axis` minimising |a[i] - b[i+s]| (2-D)."""
    n = a.shape[axis]
    m = MAX_SHIFT
    core_a = np.take(a, range(m, n - m), axis=axis)
    errs = [np.abs(core_a - np.take(b, range(m + s, n - m + s), axis=axis)).mean()
            for s in range(-m, m + 1)]
    return int(np.argmin(errs)) - m


def directional_shift(prev: np.ndarray, cur: np.ndarray, cls: int) -> float:
    """How far the picture moved from prev to cur along the swipe's axis (pixels, unsigned:
    whether the camera or the character moves, and which way, differs between games)."""
    body = slice(prev.shape[0] // 5, prev.shape[0])              # skip the HUD band at the top
    p, c = prev[body].astype(np.float32), cur[body].astype(np.float32)
    return float(abs(best_shift(p, c, axis=1 if cls in (3, 4) else 0)))


def profile(frames, starts, classes):
    out = np.zeros((len(starts), len(OFFSETS)), np.float32)
    for i, (t, cls) in enumerate(zip(starts, classes)):
        clip = np.asarray(frames[t + OFFSETS[0] - 1: t + OFFSETS[-1] + 1])
        for j in range(len(OFFSETS)):
            out[i, j] = directional_shift(clip[j], clip[j + 1], cls)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--fps", type=float, default=24.0, help="frame rate of the training videos")
    ap.add_argument("--device_latency_s", type=float, default=0.1,
                    help="capture -> inference -> injected swipe delay to predict ahead of (default 0.1 s)")
    ap.add_argument("--max_swipes", type=int, default=3000)
    a = ap.parse_args()

    meta = np.load(os.path.join(a.dataset_dir, "meta.npz"))
    T, S = int(meta["T_total"]), int(meta["img_size"])
    frames = np.memmap(os.path.join(a.dataset_dir, "frames.bin"), dtype="uint8", mode="r", shape=(T, S, S))
    mf = pd.read_csv(os.path.join(a.dataset_dir, "manifest.csv"))
    mf["sess_len"] = mf.groupby("session")["local_t"].transform("max") + 1
    margin = (mf["local_t"] + OFFSETS[0] - 1 >= 0) & (mf["local_t"] + OFFSETS[-1] < mf["sess_len"])
    rng = np.random.default_rng(0)

    sw = mf[(mf["has_action"] == 1) & margin]
    sw = sw.iloc[rng.permutation(len(sw))[: a.max_swipes]]
    if len(sw) < 20:
        raise SystemExit("too few labelled swipes with enough context")
    prof = profile(frames, sw["global_idx"].to_numpy(), sw["swipe_class"].to_numpy())

    # Baseline: random non-swipe frames, scored with random directions.
    bg = mf[(mf["has_action"] == 0) & margin]
    bg = bg.iloc[rng.permutation(len(bg))[: len(sw)]]
    base = profile(frames, bg["global_idx"].to_numpy(), rng.integers(1, 5, len(bg)))
    mu, sd = base.mean(), base.std() / math.sqrt(len(bg))

    resp = prof.mean(0)
    sem = prof.std(0) / math.sqrt(len(prof))
    thr = mu + max(4 * math.hypot(sd, sem.max()), 0.1)

    print(f"{len(sw)} swipes; picture shift along the swipe axis, frame t+o vs t+o-1 (pixels at {S}px)")
    print(f"baseline {mu:+.3f}; significance threshold {thr:+.3f}\n")
    peak = resp.max()
    for o, r in zip(OFFSETS, resp):
        bar = "#" * int(round(40 * max(r - mu, 0) / max(peak - mu, 1e-6)))
        print(f"  t{o:+3d}  {r:+.3f}  {bar}{'   <- labelled frame' if o == 0 else ''}")

    sig = np.flatnonzero(resp > thr)
    if len(sig) == 0:
        print("\nNo clear reaction found (the shift signal may not suit this game's camera).")
        return
    R = int(OFFSETS[sig[0]])
    lat = int(math.ceil(a.device_latency_s * a.fps))
    lead = max(1 - R, 0) + lat
    print(f"\nReaction onset R = {R:+d} frames ({R / a.fps * 1000:+.0f} ms) relative to the label.")
    if R <= 0:
        print("The labelled frame already shows the move: the old model learned to recognise swipes that "
              "had already happened, so it can only copy the player.")
    print(f"Train with:  --label_lead {lead} --react_delay {R}")
    print(f"  (positive = {lead} frames before the label: {max(1 - R, 0)} to get before the visible reaction "
          f"+ {lat} for ~{a.device_latency_s * 1000:.0f} ms device latency)")


if __name__ == "__main__":
    main()
