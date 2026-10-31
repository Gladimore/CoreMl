# AIPlayer — Core ML swipe player

Screen capture (ReplayKit) → exact training preprocessing → Core ML model → detection gate → synthesized swipe. Injected into the game with Sideloadly.

```
checkpoints/best.pt ──convert_to_coreml.py──▶ build/AIPlayerConfig.json
                                              build/*.mlmodel ──coremlcompiler──▶ *.mlmodelc
AIPlayer.m + AIPreprocess.c ──Theos──▶ AIPlayer.dylib
```

## Deploying

1. Put the checkpoint you want at `checkpoints/best.pt`. The converter detects which kind it is:
   - **temporal** (recommended): `train_temporal.py` / `TemporalSwipeNet`, a stack of recent frames
   - **simple**: `train_simple.py` / `SimpleSwipeAnnotator`, one grayscale frame
   - **causal** (legacy): `train_causal.py` / `CausalSwipeAnnotator`, an 8+2-frame GRU window
2. Push. CI tests the preprocessing, converts the checkpoint, checks Core ML against PyTorch, compiles everything and uploads `AIPlayer-dylib` and `AIPlayer-bundle`.
3. Unzip `AIPlayerBundle.zip` to get `AIPlayer.bundle`, which holds `AIPlayerConfig.json`, the `.mlmodelc` folder(s) and an `Info.plist`. In Sideloadly, inject both `AIPlayer.dylib` and `AIPlayer.bundle`.

The dylib looks for `AIPlayer.bundle` next to itself, in the app's root and in `Frameworks/`, then searches the app a few levels deep. If the models aren't found, the device log says where it looked (`AIPlayer` subsystem `com.aiplayer.tweak`).

To change a setting such as `fire_threshold` without rebuilding, edit `AIPlayer.bundle/AIPlayerConfig.json` before injecting.

## AIPlayerConfig.json

The converter writes this file. The dylib reads it at launch, so you can retune without rebuilding.

| key | meaning |
|---|---|
| `model_type`, `models` | which pipeline to run and which `.mlmodelc` to load. Set by the converter; don't edit. |
| `frame_offsets` | temporal only: which past ticks are stacked (e.g. `[0,1,2,4,8]`). Set by the converter; don't edit. |
| `tick_hz` | How often a frame is evaluated. **temporal / causal:** must equal the training video frame rate (the converter takes it from `--fps`). **simple:** no temporal dependence, so higher means lower latency (default 30; check `infer=` ms in the log before going to 60). |
| `fire_threshold` | Detection probability that fires a swipe. Default is the val-F1-optimal threshold saved by the fixed trainer, else the training `--det_threshold`. |
| `release_threshold` | Probability must drop below this before the next swipe can fire (one fire per burst). |
| `cooldown_s` | Minimum time between swipes (default 0.15). |
| `rearm_dir_confidence` | A *different* move predicted at least this confidently on two ticks in a row may fire before the probability has dropped (combos like LEFT→UP). 0 disables. |
| `min_dir_confidence` | Skip a swipe if the direction softmax is below this. |
| `input_gain`, `input_bias` | `gray' = gray*gain + bias` before the model. Use the values `tools/diagnose.py` suggests. |
| `compute_units` | `all` (default), `cpu_and_gpu`, or `cpu_only` (fp32, for debugging). |
| `debug_dump_frames` | If greater than 0, saves that many model inputs and predictions to `Documents/AIPlayerDump/`. |

## When it misbehaves on device

```
# 1. set "debug_dump_frames": 600, inject, play ~30 s with AI ON, copy the dump folder off the phone
python tools/diagnose.py --dump <dump folder> --checkpoint checkpoints/best.pt --dataset_dir <dataset>
```

The script reports:

- whether Core ML matches PyTorch on the exact same frames
- whether device frames look like training frames (percentiles, per-row profile, and a side-by-side PNG)
- whether the model fires more on device frames than on training frames
- the `input_gain`/`input_bias` values that correct a global brightness shift

Dumps are written as `frames.b64` (base64 text, one frame per line), so they survive apps that treat files as text. A raw binary dump gets mangled that way: every byte ≥128 becomes U+FFFD.

`python tools/test_preprocess.py` proves that `AIPreprocess.c` is bit-exact with the dataset builder. CI runs it too.

## What changed and why ("fires randomly")

**Deployment**
- **Preprocessing now matches training exactly.**
  - Training: torchvision antialiased-bilinear resize (float, then rounded to uint8), then grayscale `0.2989R + 0.587G + 0.114B` truncated to uint8.
  - Old device code: CoreImage Lanczos resize, which differed by up to ~25 grey levels at edges.
  - Old device code also used *colour-managed* rendering (DeviceRGB working space). That rewrites tone curves, which FFmpeg/NVDEC never did when the training frames were decoded.
  - Now `AIPreprocess.c` implements PyTorch's exact algorithm, tested bit-exact. CoreImage only does the YUV→RGB step, with colour management off.
- **One pipeline per model, declared by the converter.** Previously the repo could only run the causal model, and the dylib hard-coded its shapes and thresholds. A simple-model checkpoint couldn't be deployed, and a model/pipeline mismatch couldn't be detected. Model input shapes are now checked against the config at load.
- **The overlay button is kept out of the frames.** ReplayKit's in-app capture records the app's own windows too, so the "AI: ON" button appeared in every frame. While running it shrinks to a faint 22 pt dot.
- **Thresholds come from validation.** The old fixed 0.80 / 0.50 is replaced by the trainer's measured optimum.
- Buffer orientation from ReplayKit is honoured, and a debug dump plus `diagnose.py` were added.

**Training** (`training/`, or `training/sub-3-fixed.ipynb`). Retraining is needed for these.
- **The model learned to recognise swipes, not to make them (it copies the player).** `swipes.csv` labels the frame where the touch happened, and the game already shows the move starting on or right after that frame. A single-frame model therefore learns "the character is starting to move left, so LEFT". On device it fires only right after *you* swipe, in the same direction, and never on its own. The suspiciously high single-frame `dir_acc` (~0.95) is the giveaway. Fix: `--label_lead N` moves every positive N frames *before* the swipe. The still-pre-reaction frames in between are dropped, and frames showing the reaction stay negative. Measure first with `python tools/label_timing.py --dataset_dir <dataset>`, which prints the exact `--label_lead` / `--react_delay` to use. Expect a lower val F1 afterwards: the old number was inflated by this leak.
- **Label leak in augmentation (most likely root cause).** The old loader brightness/contrast-jittered *only swipe frames* (`if is_action and rand < 0.7`). The model learns "the image looks re-exposed, so swipe". Validation is never augmented, so the metrics look fine. On device, any global intensity difference against the recordings then raises the detection probability across the board. Jitter is now applied independently of the label.
- **Validation sampled with replacement.** The weighted sampler drew a different random resample every epoch. It now visits every val frame exactly once.
- **Threshold sweep each epoch.** The checkpoint stores `best_threshold`, which the converter uses.
- **UI masks were very likely transposed.** They read as `[x1,y1,x2,y2]` HUD boxes but were indexed as rows/columns, so the HUD was never masked. They are now read as x/y. Check `python training/dataset_loader_simple.py --preview_masks <dataset>` before retraining.

## Temporal model (recommended)

`model_temporal.py` and `training/train_temporal.py` use the same dataset as the simple model, so no rebuild is needed.

**Design.** Speed is a pixel-level fact: how far the track and obstacles moved since the last frame. The model stacks frames **t, t−1, t−2, t−4, t−8** as input channels, so the first convolution compares the same pixel across time. Those 1/2/4/8-tick baselines measure motion at both slow and fast speeds, with about 1/3 s of context.
- The old GRU model reduced each frame to a 128-number summary *before* comparing frames, which discarded most of that motion, and it cost 10 CNN passes per sample.
- A stride-2 first layer and a spatial head replace the full-resolution convolutions and global pooling. The spatial head keeps *where* things are: the lane, and how far down the track.
- Result: about 176M operations per prediction against 684M for the simple model (4× cheaper), and 0.85M parameters.
- On the phone it is one Core ML model with one call per tick, plus a ring of the last 9 grey frames. There is no hidden state.

**Training speed.**
- The whole dataset (9.5 GB of uint8) sits in GPU memory. A batch is one GPU gather, `frames[idx − offsets]`, so there are no DataLoader workers and no per-sample Python.
- Augmentation runs on the GPU per batch and is identical across a sample's frames. It includes simulated "held" ticks.
- It falls back to RAM or memmap automatically.
- AMP, channels_last and fused AdamW; epochs are counted in samples.

**Validation that measures play, not frames.** Every validation frame is run in order through the exact on-device gate. Each epoch prints:
- event F1: swipes caught in time *with the right direction*
- false swipes per minute
- swipes per minute compared with yours
- timing compared with your own swipes, as mean ± spread

`best.pt` is chosen on event F1, and its threshold becomes `fire_threshold`.

**Synthetic check.** In a runner where the right moment to swipe depends on speed:

| | event F1 | timing (slow / fast part of run) | timing spread | frame F1 | epoch time |
|---|---|---|---|---|---|
| simple | 0.67 | −217 / −173 ms (drifts with speed) | ±100 ms | 0.64 | ~35 s |
| temporal | 0.72 | −252 / −242 ms (consistent) | ±49 ms | 0.82 | ~10 s |

**Frame rate matters.** The device must tick at the videos' real frame rate (`--fps`, which becomes `tick_hz`). The notebook prints each video's fps. If the recordings are variable frame rate, resample them to a fixed rate before training a temporal model.

## Making the simple model more responsive

**Train for confident peaks.** Use `--label_window 3` (default) and `--action_frac 0.25` (default).
- With the old labels, exactly one frame per swipe was positive, while its near-identical neighbours were negative.
- The model therefore learned that no frame is ever clearly "the" moment. Its probability stayed low and it rarely crossed the fire threshold.
- A window of 3 frames before the swipe, all positive, gives sustained, confident peaks. The gate still fires once per peak, at the start of the window, which is also earlier.

**Calibrate the threshold on real play.** Set `debug_dump_frames` to 1200 and keep the AI on mid-run for about 40 s. Then run `diagnose.py --dataset_dir ...`. Section D replays the exact on-device gate over the dump at each threshold and recommends the `fire_threshold` that makes the AI swipe about as often as you did in the recordings. The val-F1 threshold is tuned for frame-level F1, not for play.

**Allow combos.** `rearm_dir_confidence` lets a confident new direction fire mid-burst. `cooldown_s` is now 0.15.

**Lower latency.** For the simple model, `tick_hz` 30 (or 60 if `infer=` stays under ~12 ms).

**Speed is the hard limit.** One still frame contains no motion, so it cannot see how fast the run is going. Some cues hint at it indirectly (how far down the track obstacles appear, motion blur). The only real fix is a second frame (frame + previous frame, or a diff channel). That is the smallest step toward temporality and does not need a GRU. Until then:
- `--label_lead` decides how far ahead the model acts. If it swipes too late at high speed, retrain with a larger lead (e.g. +2 frames).
- A lower `fire_threshold` also makes it act earlier within each window.

## Files

- `AIPlayer.m`: capture, tick clock, Core ML, gate, touch synthesis (unchanged), overlay
- `AIPreprocess.c/.h`: training-exact resize + grayscale + diff (plain C, unit-tested)
- `convert_to_coreml.py`: checkpoint → `.mlmodel` + `AIPlayerConfig.json`, with parity checks
- `model_temporal.py`, `model_simple.py`, `model_causal.py`: architectures (used by training and the converter)
- `tools/`: `test_preprocess.py`, `diagnose.py` (incl. threshold calibration), `label_timing.py`
- `training/`: `train_temporal.py` (fast GPU-resident trainer), fixed simple loader and trainer, and the patched notebook
