#!/usr/bin/env python3
"""
convert_to_coreml.py — Convert a CausalSwipeAnnotator checkpoint (.pt) into a
Core ML .mlpackage for streaming, single-frame-per-tick inference.

Run this on your Mac (coremltools' mlprogram/ANE toolchain requires macOS to
be fully useful, and you'll want Xcode's coremlcompiler for the next step
anyway). Requires: torch, coremltools.

    pip install torch coremltools

Usage:
    python convert_to_coreml.py \
        --checkpoint runs/causal_v1/best.pt \
        --out SwipeAnnotator.mlpackage \
        --img_size 128

`--img_size` must match what the model was trained on. The script tries to
recover it automatically from checkpoint['args']['img_size'] (train_causal.py
saves the full argparse Namespace there); pass it explicitly if that lookup
fails, e.g. because you trained with an older checkpoint format.

Output: an .mlpackage with 5 inputs / 4 outputs (see IO_SPEC below), ready to
be compiled to .mlmodelc (see the printed instructions at the end) and loaded
via the MLModel API from Swift/ObjC. Core ML does not itself produce a
.dylib — that's a step you take afterwards if you need a C/C++-callable
wrapper (see the "Getting to a .dylib" note printed at the end).
"""

import argparse
import sys

import torch
import torch.nn as nn

from model_causal import build_causal_model, CausalSwipeAnnotator


IO_SPEC = """
Inputs:
  fast_frame  (1, 2, H, W) float32   current frame's (frame, diff) pair
  slow_frame  (1, 2, H, W) float32   slow-branch sample; pass zeros when has_slow=0
  has_slow    (1,)          float32   1.0 if slow_frame is a real new sample this tick, else 0.0
  h_fast_in   (fast_layers, 1, hidden)       carried GRU state, fast branch
  h_slow_in   (slow_layers, 1, slow_hidden)  carried GRU state, slow branch

Outputs:
  det_logits  (1,)    sigmoid() this for the detection probability
  dir_logits  (1, 4)  softmax() this for [UP, DOWN, LEFT, RIGHT] probabilities
  h_fast_out  (fast_layers, 1, hidden)       feed into next call's h_fast_in
  h_slow_out  (slow_layers, 1, slow_hidden)  feed into next call's h_slow_in

Session start: zero-init h_fast_in / h_slow_in (matches model.init_state()).
Every subsequent call: feed back h_fast_out / h_slow_out from the previous call.
"""


class StepWrapper(nn.Module):
    """
    Traceable single-step wrapper around CausalSwipeAnnotator.step().

    torch.jit.trace bakes in whatever Python control flow it happens to
    execute during tracing -- it can NOT export a real runtime branch. The
    original step()'s `if slow_frame is not None` would silently become a
    permanent constant (always-has-slow or always-no-slow) baked into the
    traced graph, depending on which one you traced with. We avoid that by
    using a `has_slow` tensor to blend between "new slow state" and "carried
    slow state" arithmetically, so the exported model has ONE fixed graph
    that behaves correctly at runtime for either case, selected each call by
    an input tensor rather than by which Python branch got traced.
    """

    def __init__(self, model: CausalSwipeAnnotator):
        super().__init__()
        self.model = model

    def forward(self, fast_frame, slow_frame, has_slow, h_fast, h_slow):
        fast_feat = self.model._cnn_features(fast_frame).unsqueeze(1)
        _, h_fast_new = self.model.fast_gru(fast_feat, h_fast)

        slow_feat = self.model._cnn_features(slow_frame).unsqueeze(1)
        _, h_slow_candidate = self.model.slow_gru(slow_feat, h_slow)

        mask = has_slow.view(1, 1, 1)
        h_slow_new = mask * h_slow_candidate + (1.0 - mask) * h_slow

        fused = torch.cat([h_fast_new[-1], h_slow_new[-1]], dim=-1)
        fused = self.model.drop(fused)  # no-op in eval() anyway

        det_logits = self.model.det_head(fused).squeeze(-1)
        dir_logits = self.model.dir_head(fused)
        return det_logits, dir_logits, h_fast_new, h_slow_new


def recover_img_size(checkpoint: dict, cli_value: int) -> int:
    if cli_value is not None:
        return cli_value
    args = checkpoint.get('args', {})
    for key in ('img_size', 'image_size', 'frame_size'):
        if key in args:
            print(f"[info] img_size recovered from checkpoint['args']['{key}'] = {args[key]}")
            return int(args[key])
    print(
        "[error] Could not find img_size in checkpoint['args']. "
        "Pass it explicitly with --img_size (must match your training data's H=W).",
        file=sys.stderr,
    )
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint (e.g. best.pt)')
    ap.add_argument('--out', default='SwipeAnnotator.mlpackage', help='Output .mlpackage path')
    ap.add_argument('--img_size', type=int, default=None,
                     help='Frame H=W. Auto-recovered from checkpoint args if omitted.')
    ap.add_argument('--min_ios', default='iOS26',
                     choices=['iOS16', 'iOS17', 'iOS18', 'iOS26'],
                     help='minimum_deployment_target. Default iOS26 (matches a 26.5.2 device); '
                          'lower this only if you also need older-device support.')
    args = ap.parse_args()

    print(f"[1/5] Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    if 'model_state' not in checkpoint or 'arch' not in checkpoint:
        print(f"[error] '{args.checkpoint}' doesn't look like a train_causal.py checkpoint "
              f"(missing 'model_state' / 'arch' keys). Found keys: {list(checkpoint.keys())}",
              file=sys.stderr)
        sys.exit(1)

    img_size = recover_img_size(checkpoint, args.img_size)
    arch = checkpoint['arch']
    print(f"[info] arch = {arch}")
    print(f"[info] checkpoint epoch={checkpoint.get('epoch')} val_f1={checkpoint.get('val_f1')}")

    print("[2/5] Building model and loading weights")
    model = build_causal_model(arch)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    wrapper = StepWrapper(model).eval()

    B, H, W = 1, img_size, img_size
    example = (
        torch.zeros(B, 2, H, W),                                   # fast_frame
        torch.zeros(B, 2, H, W),                                   # slow_frame
        torch.zeros(1),                                            # has_slow
        torch.zeros(model.fast_layers, B, model.hidden),           # h_fast_in
        torch.zeros(model.slow_layers, B, model.slow_hidden),      # h_slow_in
    )

    print("[3/5] Tracing with torch.jit.trace")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, check_trace=True)

    print("[4/5] Converting with coremltools")
    import coremltools as ct

    target = getattr(ct.target, args.min_ios)
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="fast_frame", shape=example[0].shape, dtype=float),
            ct.TensorType(name="slow_frame", shape=example[1].shape, dtype=float),
            ct.TensorType(name="has_slow",   shape=example[2].shape, dtype=float),
            ct.TensorType(name="h_fast_in",  shape=example[3].shape, dtype=float),
            ct.TensorType(name="h_slow_in",  shape=example[4].shape, dtype=float),
        ],
        outputs=[
            ct.TensorType(name="det_logits"),
            ct.TensorType(name="dir_logits"),
            ct.TensorType(name="h_fast_out"),
            ct.TensorType(name="h_slow_out"),
        ],
        minimum_deployment_target=target,
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.ALL,  # let Core ML pick ANE/GPU/CPU per-op
    )

    mlmodel.short_description = "CausalSwipeAnnotator — streaming step() inference"
    mlmodel.input_description["fast_frame"] = "Current frame (frame, diff), normalized 0-255 uint8 range"
    mlmodel.input_description["has_slow"] = "1.0 if slow_frame is a genuine new sample this tick, else 0.0"
    mlmodel.output_description["det_logits"] = "sigmoid() -> detection probability for current frame"
    mlmodel.output_description["dir_logits"] = "softmax() -> [UP, DOWN, LEFT, RIGHT] probabilities"

    print(f"[5/5] Saving to {args.out}")
    mlmodel.save(args.out)

    print(IO_SPEC)
    print(f"""
Next steps
----------
1) Sanity check by loading it back:
     python -c "import coremltools as ct; m = ct.models.MLModel('{args.out}'); print(m)"

2) Compile ahead-of-time for deployment (produces a .mlmodelc bundle):
     xcrun coremlcompiler compile {args.out} ./build/

3) Load at runtime via the MLModel API (Swift/ObjC) from your app or tweak.
   Core ML does not export a .dylib directly -- if you need a C/C++-callable
   dynamic library (e.g. calling from a non-Swift harness), write a thin
   Objective-C++ shim that loads the .mlmodelc via MLModel and exposes a C
   API, then compile THAT shim as your .dylib.

4) VALIDATE step() vs forward() parity before trusting this in production
   (see model_causal.py's docstring caveat) -- carrying hidden state
   indefinitely in step()/this export is an approximation of forward()'s
   fixed 8-frame window, not a mathematical identity. Compare traced-model
   logits against forward()-produced logits on a held-out session.
""")


if __name__ == '__main__':
    main()
