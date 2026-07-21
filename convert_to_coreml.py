#!/usr/bin/env python3
"""
convert_to_coreml.py — Convert a CausalSwipeAnnotator checkpoint (.pt) into a
Core ML .mlpackage for streaming, single-frame-per-tick inference.

Run this on your Mac (coremltools' mlprogram/ANE toolchain requires macOS to
be fully useful, and you'll want Xcode's coremlcompiler for the next step
anyway). Requires: torch, coremltools.

    pip install torch==2.7.0 coremltools

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

WHY THIS FILE DOESN'T JUST CALL self.model.fast_gru()/slow_gru() DIRECTLY
---------------------------------------------------------------------------
Calling nn.GRU's own forward() -- even with a real sequence length of 1,
which is all step()-based streaming ever uses -- makes coremltools' PyTorch
frontend lower it to a MIL `while_loop` containing `gather`/`scatter` ops
(this is how coremltools translates GRU/LSTM in general: as an explicit
step-through-the-sequence loop, regardless of whether the traced sequence
length happens to be 1). Core ML's BNNS Graph backend doesn't support that
opset, so `xcrun coremlcompiler compile` fails with "Unsupported opset for
gather op" -- and does so *silently*: it still exits 0 and still produces a
SwipeAnnotator.mlmodelc directory, just one that's missing Manifest.json and
can't actually be loaded. (Confirmed by inspecting the raw MIL text of a
converted model: the gather/scatter/while_loop trio lives entirely inside
coremltools' GRU lowering, not anywhere in this file's own code -- two
earlier attempts to fix this by changing how the *final* hidden state was
indexed [`h_fast_new[-1]` -> `.narrow()` -> `h_fast_new[fast_layers - 1]`]
had zero effect on the error for exactly this reason: neither one was ever
the actual source.)

The fix used below: `manual_gru_step()` reimplements the standard GRU update
equations by hand, using nn.GRU's own trained weight/bias tensors, called
once per layer inside a plain Python for-loop. torch.jit.trace fully unrolls
a Python loop over a small static range into straight-line ops -- there is
no sequence dimension left to loop over at the MIL level at all, so no
while_loop/gather/scatter can appear. Verified bit-exact (max abs diff
~2e-7, i.e. float32 rounding noise) against nn.GRU's own forward() on
identical weights/inputs -- see the assertion in build_and_trace() below,
which re-checks this against your actual checkpoint every time you run this
script, not just a one-off synthetic test.
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


def manual_gru_step(x: torch.Tensor, h: torch.Tensor, gru: nn.GRU) -> torch.Tensor:
    """
    Single-step, multi-layer GRU forward using `gru`'s own trained weights --
    with no internal sequence loop, since step()-based streaming inference
    only ever processes one timestep at a time. See the module docstring
    for why this exists instead of just calling gru(x, h).

    Implements PyTorch's own documented GRU equations (matches nn.GRU
    exactly -- this is not an approximation):
        r = sigmoid(W_ir x + b_ir + W_hr h + b_hr)
        z = sigmoid(W_iz x + b_iz + W_hz h + b_hz)
        n = tanh(W_in x + b_in + r * (W_hn h + b_hn))
        h' = (1 - z) * n + z * h
    where weight_ih_l{k} stacks [W_ir; W_iz; W_in] and weight_hh_l{k} stacks
    [W_hr; W_hz; W_hn] along dim 0 -- PyTorch's actual internal GRU layout.

    x: (B, input_size)        h: (num_layers, B, hidden)
    returns: h_new (num_layers, B, hidden), one row per layer's new state.
    """
    num_layers = h.shape[0]
    layer_input = x
    new_states = []
    for layer in range(num_layers):
        w_ih = getattr(gru, f'weight_ih_l{layer}')
        w_hh = getattr(gru, f'weight_hh_l{layer}')
        b_ih = getattr(gru, f'bias_ih_l{layer}')
        b_hh = getattr(gru, f'bias_hh_l{layer}')
        h_layer = h[layer]
        hidden = h_layer.shape[-1]

        gi = layer_input @ w_ih.t() + b_ih
        gh = h_layer @ w_hh.t() + b_hh
        i_r, i_z, i_n = gi.split(hidden, dim=-1)
        h_r, h_z, h_n = gh.split(hidden, dim=-1)
        r = torch.sigmoid(i_r + h_r)
        z = torch.sigmoid(i_z + h_z)
        n = torch.tanh(i_n + r * h_n)
        h_new_layer = (1 - z) * n + z * h_layer

        new_states.append(h_new_layer)
        layer_input = h_new_layer  # next layer's input is this layer's output

    return torch.stack(new_states, dim=0)


def assert_manual_gru_matches(model: CausalSwipeAnnotator, atol: float = 1e-5):
    """
    Re-verifies manual_gru_step() against nn.GRU's own forward() using THIS
    checkpoint's actual trained weights, every time this script runs -- not
    just a one-off synthetic test during development. Cheap (a handful of
    random-input forward passes) and it's the one check standing between
    "the export is mathematically faithful" and "silently wrong
    predictions on-device", so it isn't optional.
    """
    torch.manual_seed(0)
    for gru, num_layers, hidden in (
        (model.fast_gru, model.fast_layers, model.hidden),
        (model.slow_gru, model.slow_layers, model.slow_hidden),
    ):
        x = torch.randn(1, 128)
        h0 = torch.randn(num_layers, 1, hidden)
        with torch.no_grad():
            _, h_ref = gru(x.unsqueeze(1), h0)
            h_manual = manual_gru_step(x, h0, gru)
        max_diff = (h_ref - h_manual).abs().max().item()
        if not torch.allclose(h_ref, h_manual, atol=atol):
            print(f"[error] manual_gru_step diverges from nn.GRU by {max_diff} "
                  f"(tolerance {atol}) -- DO NOT trust the exported model. "
                  f"This should only fail if manual_gru_step's math or gate "
                  f"ordering has been changed incorrectly.", file=sys.stderr)
            sys.exit(1)
        print(f"[info] manual_gru_step verified against nn.GRU (max diff {max_diff:.2e})")


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

    GRU steps go through manual_gru_step() rather than calling
    self.model.fast_gru()/slow_gru() directly -- see the module docstring
    for why (coremltools' generic GRU->MIL lowering emits a while_loop with
    gather/scatter that Core ML's BNNS Graph backend can't compile, even
    though our traced sequence length is always exactly 1).
    """

    def __init__(self, model: CausalSwipeAnnotator):
        super().__init__()
        self.model = model

    def forward(self, fast_frame, slow_frame, has_slow, h_fast, h_slow):
        fast_feat = self.model._cnn_features(fast_frame)   # (B, 128)
        h_fast_new = manual_gru_step(fast_feat, h_fast, self.model.fast_gru)

        slow_feat = self.model._cnn_features(slow_frame)   # (B, 128)
        h_slow_candidate = manual_gru_step(slow_feat, h_slow, self.model.slow_gru)

        mask = has_slow.view(1, 1, 1)
        h_slow_new = mask * h_slow_candidate + (1.0 - mask) * h_slow

        # Plain positive-int indexing on a statically-known dim -- traces to
        # aten::select, not a gather. (Not the source of the original bug,
        # but no reason to use a negative index either.)
        fast_layers = self.model.fast_layers
        slow_layers = self.model.slow_layers
        h_fast_last = h_fast_new[fast_layers - 1]
        h_slow_last = h_slow_new[slow_layers - 1]

        fused = torch.cat([h_fast_last, h_slow_last], dim=-1)
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

    print("[2.5/5] Verifying manual_gru_step against nn.GRU on this checkpoint's weights")
    assert_manual_gru_matches(model)

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

    # Belt-and-suspenders: fail here, with a clear message, rather than
    # discovering a gather/while_loop op survived some future refactor only
    # when coremlcompiler chokes on it three steps downstream in CI.
    mil_text = str(mlmodel._mil_program)
    banned_ops = [op for op in ("gather(", "while_loop(", "scatter(") if op in mil_text]
    if banned_ops:
        print(f"[error] Exported MIL graph still contains {banned_ops} -- "
              f"these are exactly the ops that break BNNS Graph compilation. "
              f"Something in this file changed to call an RNN op directly "
              f"again instead of going through manual_gru_step().",
              file=sys.stderr)
        sys.exit(1)
    print("[info] Confirmed: no gather/while_loop/scatter ops in the exported graph")

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
