#!/usr/bin/env python3
"""
test_preprocess.py — prove that AIPreprocess.c reproduces the training pipeline.

Compiles AIPreprocess.c into a temporary shared library and compares it with the
exact torchvision ops the dataset builder ran (resize antialias=True on float32,
round to uint8, rgb_to_grayscale). Also checks AIDiffValue against the integer
formula of the causal builder.

    pip install torch torchvision
    python tools/test_preprocess.py

Needs a C compiler (cc/clang/gcc). Exits non-zero on mismatch.
"""

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torchvision.transforms.v2.functional as TF

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = 128


class Axis(ctypes.Structure):
    _fields_ = [("inSize", ctypes.c_int), ("outSize", ctypes.c_int), ("maxTaps", ctypes.c_int),
                ("start", ctypes.POINTER(ctypes.c_int)), ("count", ctypes.POINTER(ctypes.c_int)),
                ("weights", ctypes.POINTER(ctypes.c_float))]


class Resampler(ctypes.Structure):
    _fields_ = [("x", Axis), ("y", Axis), ("rowTmp", ctypes.POINTER(ctypes.c_float))]


def build_lib(tmp: str):
    cc = shutil.which("cc") or shutil.which("clang") or shutil.which("gcc")
    if not cc:
        sys.exit("no C compiler found")
    so = os.path.join(tmp, "libaipre.so")
    subprocess.check_call([cc, "-O2", "-shared", "-fPIC", "-o", so,
                           os.path.join(ROOT, "AIPreprocess.c"), "-lm"])
    lib = ctypes.CDLL(so)
    lib.AIResamplerInit.argtypes = [ctypes.POINTER(Resampler)] + [ctypes.c_int] * 4
    lib.AIResamplerFree.argtypes = [ctypes.POINTER(Resampler)]
    lib.AIResizeToGray.argtypes = [ctypes.POINTER(Resampler), ctypes.c_void_p, ctypes.c_size_t,
                                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.AIDiffValue.argtypes = [ctypes.c_uint8, ctypes.c_uint8]
    lib.AIDiffValue.restype = ctypes.c_uint8
    return lib


def reference(rgb_hwc: np.ndarray) -> np.ndarray:
    """What build_dataset_*_gpu.py computes (resize and grayscale on uint8 tensors)."""
    t = torch.from_numpy(rgb_hwc).permute(2, 0, 1).unsqueeze(0)      # (1, 3, H, W) uint8
    # On GPU torchvision casts uint8 -> float32, interpolates, rounds; force the same path on CPU.
    f = torch.nn.functional.interpolate(t.float(), size=[OUT, OUT], mode="bilinear",
                                        align_corners=False, antialias=True)
    t = f.round_().to(torch.uint8)
    return TF.rgb_to_grayscale(t).squeeze().numpy()


def natural_image(h: int, w: int, seed: int) -> np.ndarray:
    """Smooth gradients + hard edges + noise: closer to game frames than pure noise."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.zeros((h, w, 3), np.float32)
    for c in range(3):
        img[..., c] = 127 + 100 * np.sin(xx / rng.uniform(20, 200) + c) * np.cos(yy / rng.uniform(20, 300))
    for _ in range(40):                                               # hard-edged blocks
        y0, x0 = rng.integers(0, h - 50), rng.integers(0, w - 50)
        img[y0:y0 + rng.integers(10, 400), x0:x0 + rng.integers(10, 300)] = rng.uniform(0, 255, 3)
    img += rng.normal(0, 12, img.shape)
    img[: h // 20] = 255                                                # saturated band
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    tmp = tempfile.mkdtemp()
    try:
        lib = build_lib(tmp)
        worst = 0
        for (h, w), seed in (((1920, 886), 0), ((2556, 1179), 1), ((1334, 750), 2), ((100, 90), 3)):
            rgb = natural_image(h, w, seed)
            ref = reference(rgb)

            rs = Resampler()
            assert lib.AIResamplerInit(ctypes.byref(rs), w, h, OUT, OUT) == 0
            for name, order in (("RGBA", (0, 1, 2)), ("BGRA", (2, 1, 0))):
                px = np.zeros((h, w, 4), np.uint8)
                px[..., order[0]], px[..., order[1]], px[..., order[2]] = rgb[..., 0], rgb[..., 1], rgb[..., 2]
                px[..., 3] = 255
                out = np.zeros((OUT, OUT), np.uint8)
                lib.AIResizeToGray(ctypes.byref(rs), px.ctypes.data, w * 4, *order, out.ctypes.data)
                d = np.abs(out.astype(int) - ref.astype(int))
                worst = max(worst, int(d.max()))
                print(f"{w}x{h} {name}: max |diff| = {d.max()}  pixels off by 1: {(d == 1).mean():.4%}")
            lib.AIResamplerFree(ctypes.byref(rs))

        # Off-by-one at float32 rounding ties is expected (CPU vs GPU float order); more is a bug.
        if worst > 1:
            sys.exit(f"[FAIL] resize/grayscale differs from training by up to {worst} levels")

        cur = np.arange(256).repeat(256)
        prev = np.tile(np.arange(256), 256)
        ref_diff = np.clip((cur - prev) * 127 // 255 + 127, 0, 255)
        got = np.array([lib.AIDiffValue(int(a), int(b)) for a, b in zip(cur, prev)])
        if not np.array_equal(got, ref_diff):
            sys.exit("[FAIL] AIDiffValue differs from the causal builder's formula")
        print("diff LUT: exact for all 65536 pairs")
        print("[OK] AIPreprocess.c matches training preprocessing")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
