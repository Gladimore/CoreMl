# AIPlayer — Core ML swipe player

Screen capture (ReplayKit) → exact training preprocessing → Core ML model → detection gate → synthesized swipe. Injected into the game with Sideloadly.

```
checkpoints/best.pt ──convert_to_coreml.py──▶ build/AIPlayerConfig.json
                                              build/*.mlmodel ──coremlcompiler──▶ *.mlmodelc
AIPlayer.m + AIPreprocess.c ──Theos──▶ AIPlayer.dylib
```

## Deploying

1. Put the checkpoint you want at `checkpoints/best.pt`. Both kinds work, and the converter detects which one it is:
   - **simple**: `train_simple.py` / `SimpleSwipeAnnotator`, one grayscale frame
   - **causal**: `train_causal.py` / `CausalSwipeAnnotator`, an 8+2-frame GRU window
2. Push. CI tests the preprocessing, converts the checkpoint, checks Core ML against PyTorch, compiles everything and uploads `AIPlayer-dylib` and `AIPlayer-bundle`.
3. Unzip `AIPlayerBundle.zip` to get `AIPlayer.bundle`, which holds `AIPlayerConfig.json`, the `.mlmodelc` folder(s) and an `Info.plist`. In Sideloadly, inject both `AIPlayer.dylib` and `AIPlayer.bundle`.

The dylib looks for `AIPlayer.bundle` next to itself, in the app's root and in `Frameworks/`, then searches the app a few levels deep. If the models aren't found, the device log says where it looked (`AIPlayer` subsystem `com.aiplayer.tweak`).

To change a setting such as `fire_threshold` without rebuilding, edit `AIPlayer.bundle/AIPlayerConfig.json` before injecting.

## AIPlayerConfig.json

The converter writes this file. The dylib reads it at launch, so you can retune without rebuilding.

| key | meaning |
|---|---|
| `model_type`, `models` | which pipeline to run and which `.mlmodelc` to load. Set by the converter; don't edit. |
| `tick_hz` | Frame rate of the **training videos**. The causal model depends on it; the simple model barely does. |
| `fire_threshold` | Detection probability that fires a swipe. Default is the val-F1-optimal threshold saved by the fixed trainer, else the training `--det_threshold`. |
| `release_threshold` | Probability must drop below this before the next swipe can fire (one fire per burst). |
| `cooldown_s` | Minimum time between swipes. |
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
- **Label leak in augmentation (most likely root cause).** The old loader brightness/contrast-jittered *only swipe frames* (`if is_action and rand < 0.7`). The model learns "the image looks re-exposed, so swipe". Validation is never augmented, so the metrics look fine. On device, any global intensity difference against the recordings then raises the detection probability across the board. Jitter is now applied independently of the label.
- **Validation sampled with replacement.** The weighted sampler drew a different random resample every epoch. It now visits every val frame exactly once.
- **Threshold sweep each epoch.** The checkpoint stores `best_threshold`, which the converter uses.
- **UI masks were very likely transposed.** They read as `[x1,y1,x2,y2]` HUD boxes but were indexed as rows/columns, so the HUD was never masked. They are now read as x/y. Check `python training/dataset_loader_simple.py --preview_masks <dataset>` before retraining.

## Files

- `AIPlayer.m`: capture, tick clock, Core ML, gate, touch synthesis (unchanged), overlay
- `AIPreprocess.c/.h`: training-exact resize + grayscale + diff (plain C, unit-tested)
- `convert_to_coreml.py`: checkpoint → `.mlmodel` + `AIPlayerConfig.json`, with parity checks
- `model_simple.py`, `model_causal.py`: architectures for conversion
- `tools/`: `test_preprocess.py`, `diagnose.py`
- `training/`: fixed loader and trainer, and the patched notebook
