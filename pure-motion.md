# Pure Motion -- Lip Syncer Integration

## What this is

Pure Motion is a hybrid lip-sync refinement feature added to the existing `lip_syncer` processor in FaceFusion 3.6.0. It replaces the standard "paint pixels onto the mouth" approach (wav2lip / edtalk) with a two-stage pipeline:

1. Run the normal lip-sync model on a **neutral face template** to produce an audio-driven mouth shape.
2. Extract that mouth shape as a LivePortrait **expression vector**, then re-render the real target face with that expression blended in.

The result is audio-driven mouth motion that preserves the target face's identity and texture rather than compositing a synthetic mouth patch.

The feature is gated by a single slider: **LIP SYNCER PURE MOTION**. When set to 0 (default), the processor behaves identically to stock FaceFusion. When set above 0, the LivePortrait refinement path activates. Values above 1.0 over-drive the expression for more pronounced mouth movement.

## Current handoff status

The initial port is in place and the repo is now past the first major runtime blocker:

- Pure Motion is wired into the `lip_syncer` processor and exposed in both CLI and Gradio UI.
- The original stale inference-pool bug that produced `AttributeError: 'NoneType' object has no attribute 'run'` from `motion_extractor` has been fixed in `facefusion/processors/modules/lip_syncer/core.py`.
- The UI now has a **SAVE AS DEFAULT** button that writes the current reusable settings back to `facefusion.ini`, including `lip_syncer_pure_motion`.

What remains is refinement work, not first-pass integration:

- visual tuning of expression blending
- validating model/provider combinations on real footage
- hardening edge cases like template face detection and execution-provider variance
- improving the repeatability of test profiles and default settings

## Origin

This work was ported from `pure_motion_2.patch` (still in the repo root for reference). That patch targeted an older FaceFusion layout with a monolithic `facefusion/processors/modules/lip_syncer.py` file. The current repo uses a package-based layout (`lip_syncer/core.py`, `choices.py`, `types.py`, `locales.py`), so the patch was manually adapted to the current conventions rather than applied directly.

## Architecture

```
sync_lip(target_face, audio_frame, vision_frame)
  |
  |-- warp face crop to FFHQ 512x512
  |-- create occlusion masks
  |
  |-- if pure_motion > 0:
  |     |-- extract_template_expression(audio_frame)
  |     |     |-- load face_template.npy (LRU-cached)
  |     |     |-- run lip-sync model (wav2lip/edtalk) on template
  |     |     |-- run motion_extractor on result -> expression vector
  |     |     |-- scale expression by pure_motion factor
  |     |
  |     |-- prepare_refine_frame (resize to 256x256, normalize)
  |     |-- forward_extract_feature -> feature volume
  |     |-- forward_extract_motion -> pitch/yaw/roll/scale/translation/expression/motion_points
  |     |-- calculate target motion points (original expression)
  |     |-- blend template expression into target expression
  |     |-- calculate source motion points (blended expression)
  |     |-- forward_stitch_motion_points
  |     |-- forward_generate_frame -> new face crop
  |     |-- normalize_refine_frame (back to 512x512 uint8)
  |     |-- box mask
  |
  |-- else (standard path):
  |     |-- wav2lip or edtalk on the face crop directly
  |     |-- area mask (wav2lip) or box mask (edtalk)
  |
  |-- combine masks, paste_back into original frame
```

### Key design decisions

- The lip-sync model is still used even in pure-motion mode, but it runs on a static neutral template face rather than the real target. This converts it from a pixel painter into an expression driver.
- Expression blending uses hardcoded per-index weights (indices 6, 12, 14, 17, 19, 20 of the 21-point LivePortrait expression tensor). These control how much of the audio-driven mouth shape transfers vs how much of the target's original expression is preserved.
- The inference pool is shared: when pure_motion > 0, both the lip-syncer ONNX session and all six LivePortrait ONNX sessions live in the same pool.
- The original port had a cache-key bug: the inference context only used the lip-syncer model name, so a previously created lip-sync-only pool could be reused when Pure Motion later requested LivePortrait sessions. This surfaced as `motion_extractor = None` and then `AttributeError: 'NoneType' object has no attribute 'run'`.
- The current fix gives Pure Motion its own pool identity by appending `'live_portrait'` to the `model_names` context when `has_pure_motion()` is true, and `clear_inference_pool()` now clears both `[lip_syncer_model]` and `[lip_syncer_model, 'live_portrait']`.

## Files changed (vs FaceFusion 3.6.0 base)

### `facefusion/processors/modules/lip_syncer/core.py`
The main logic file. ~340 lines added. Contains:
- Extended `create_static_model_set()` with `live_portrait` (6 sub-models) and `face_template` entries
- `collect_model_downloads()` -- builds combined DownloadSet for lip-syncer + conditionally LivePortrait models
- Modified `get_inference_pool()` to use `collect_model_downloads()` instead of just the lip-syncer sources
- `get_inference_pool_model_names()` -- adds `'live_portrait'` to the inference context when Pure Motion is active
- `get_static_face_template()` -- LRU-cached loader for `face_template.npy`, runs face detection to get bounding box
- `has_pure_motion()` -- checks `lip_syncer_pure_motion > 0`
- Refactored `sync_lip()` into branching logic with `process_standard_lip_sync()` and `process_live_portrait_motion()`
- `apply_lip_syncer()` -- shared helper that runs the lip-sync model on any crop + bounding box
- `extract_template_expression()` -- the core trick: lip-sync on template, motion_extract, scale
- `create_blended_expression()` / `blend_expression()` -- per-index expression blending
- `calculate_target_motion_points()` / `calculate_source_motion_points()` -- motion point math
- LivePortrait forwards: `forward_extract_feature()`, `forward_extract_motion()`, `forward_stitch_motion_points()`, `forward_generate_frame()`
- `prepare_refine_frame()` / `normalize_refine_frame()` -- 256x256 input prep and 512x512 output denorm
- `resize_bounding_box()` -- helper for wav2lip bounding box adjustment
- Updated `register_args()`, `apply_args()`, `pre_check()` for the new `--lip-syncer-pure-motion` option
- `clear_inference_pool()` now clears both the standard and Pure Motion pool variants to avoid stale session reuse

### `facefusion/processors/modules/lip_syncer/choices.py`
Added: `lip_syncer_pure_motion_range : Sequence[float] = create_float_range(0.0, 1.5, 0.25)`

### `facefusion/processors/modules/lip_syncer/locales.py`
Added: `help.pure_motion` and `uis.pure_motion_slider` locale strings.

### `facefusion/uis/components/lip_syncer_options.py`
Added: `LIP_SYNCER_PURE_MOTION_SLIDER` global, rendered between model dropdown and weight slider. Change handler calls `clear_inference_pool()` then updates state. `remote_update()` now returns 3 components.

### `facefusion/uis/components/preview.py`
Added: `'lip_syncer_pure_motion_slider'` to the list of sliders that trigger preview re-render on release.

### `facefusion/uis/types.py`
Added: `'lip_syncer_pure_motion_slider'` to the `ComponentName` literal.

### `facefusion/config.py`
Added `save_defaults()` plus a section/key map for serializing the current reusable UI state back into `facefusion.ini`. This now includes `lip_syncer_pure_motion`.

### `facefusion/uis/components/common_options.py`
Added a **SAVE AS DEFAULT** button and status textbox in the existing options block. This writes the current defaults to the active config file path.

### `facefusion/locales.py`
Added UI strings for `SAVE AS DEFAULT` success/failure messaging.

### `facefusion.ini`
Added the missing `lip_syncer_pure_motion =` entry so the config template and the runtime args are aligned.

## External dependencies

### Face template
- Source: https://huggingface.co/bluefoxcreation/Templates/tree/main/face-template
- Files: `face_template.hash` (8 bytes), `face_template.npy` (787 KB)
- Downloaded to: `.assets/templates/face_template.hash` and `.assets/templates/face_template.npy`
- Downloaded automatically on first `pre_check()` via FaceFusion's `conditional_download_hashes` / `conditional_download_sources`

### LivePortrait models
Same ONNX models already used by `face_editor` and `expression_restorer` processors:
- `live_portrait_feature_extractor.onnx`
- `live_portrait_motion_extractor.onnx`
- `live_portrait_eye_retargeter.onnx`
- `live_portrait_lip_retargeter.onnx`
- `live_portrait_stitcher.onnx`
- `live_portrait_generator.onnx`

All stored in `.assets/models/`. If you've already used face_editor or expression_restorer, these are already downloaded.

## How to test

### Prerequisites
- Python environment with FaceFusion 3.6.0 dependencies installed
- A target video with a visible face
- A source audio file (MP3 or WAV)

### Gradio UI test
```bash
python facefusion.py run
```
1. In the **processors** checkbox, select `lip_syncer`
2. Set a source audio file
3. Set a target video
4. The **LIP SYNCER PURE MOTION** slider appears between the model dropdown and weight slider
5. Set it to a value like 0.75 or 1.0
6. The preview should update showing audio-driven mouth motion
7. Run full video processing to verify output

### Gradio default-save test
1. Change a few settings in the UI, including `LIP SYNCER PURE MOTION`
2. Click **SAVE AS DEFAULT**
3. Confirm the success message appears
4. Confirm `facefusion.ini` now contains the updated values, including `lip_syncer_pure_motion`
5. Restart the app and verify those values load back into the UI

### CLI test
```bash
python facefusion.py headless-run \
  --processors lip_syncer \
  -s /path/to/audio.mp3 \
  -t /path/to/target.mp4 \
  -o /path/to/output.mp4 \
  --lip-syncer-pure-motion 1.0 \
  --trim-frame-end 30
```

### What to verify
- With `pure_motion = 0`: output should be identical to stock FaceFusion lip sync
- With `pure_motion > 0`: mouth should move with audio but face texture/identity should be better preserved
- With `pure_motion > 1.0`: expression should be exaggerated (over-driven)
- Toggling between 0 and non-zero should work without errors (inference pool is cleared and rebuilt)
- Preview first, then run full output, and also do the reverse order. The stale-pool bug originally appeared when one context reused an incomplete pool built by another.
- First run with pure_motion > 0 will trigger model downloads (~1.5 GB total for LivePortrait + template)
- Saving defaults from the UI should persist `lip_syncer_pure_motion` and reload correctly on restart

## Known risks and tuning areas

### Expression blending weights
The `create_blended_expression()` function uses hardcoded per-index blend factors. These were ported directly from the original patch and are empirical. The indices correspond to specific facial control points in LivePortrait's 21-point expression tensor:
- Index 6: general face shape (0.5, 0.5, 0.5)
- Index 12: mid-face (0.5, 0.5, 0.5)
- Index 14: jaw area (0.6, 0.7, 0.7)
- Index 17: lower face (0.5, 0.8, 0.7)
- Index 19: lip opening -- dynamically scaled by detected lip openness (0.5, 0.9-1.5, 0.85)
- Index 20: chin area (0.5, 0.6, 0.7)

These may need visual tuning for different face types or audio characteristics.

### Memory
When pure_motion is active, the inference pool holds 7 ONNX sessions (1 lip-syncer + 6 LivePortrait). This is significant GPU memory usage. The existing `video_memory_strategy` setting handles cleanup between frames if set to `strict` or `moderate`.

### Inference pool cache key
This was a real bug in the first port. The cache key was built from `model_names` using only the lip-syncer model name, not the Pure Motion model set. That allowed a stale pool without LivePortrait sessions to be reused, leading to `motion_extractor` being `None`.

Current mitigation:
- `get_inference_pool_model_names()` adds `'live_portrait'` to the context when Pure Motion is active
- `clear_inference_pool()` clears both the standard and Pure Motion pool variants

If future code changes alter the model set again, the inference context must continue to reflect that. Do not assume `model_source_set` alone is enough to make the pool unique.

### Template face detection
`get_static_face_template()` loads `face_template.npy` and runs FaceFusion's face detector on it to find the bounding box. If the face detector fails on the template (wrong settings, score threshold too high), this will crash with an `AttributeError` on `None.bounding_box`. The function is LRU-cached so this only happens once per session.

## Recommended next refinement tasks

1. Tune `create_blended_expression()` weights against a small visual benchmark set.
2. Compare `wav2lip_gan_96`, `wav2lip_96`, and `edtalk_256` specifically in Pure Motion mode rather than standard lip-sync mode.
3. Validate execution-provider behavior with `cuda` versus `cuda tensorrt` on Windows/NVIDIA systems; keep `cuda` as the stability baseline.
4. Harden `get_static_face_template()` so template detection failure raises a useful error instead of crashing on `None.bounding_box`.
5. Decide whether the Save-as-Default flow should stay in `common_options.py` or move to a more explicit settings/config section later.

## Suggested baseline test profile

For high-quality Pure Motion testing on a high-VRAM NVIDIA system, the current preferred baseline is:

- `processors = lip_syncer`
- `lip_syncer_model = wav2lip_gan_96`
- `lip_syncer_pure_motion = 1.0`
- `lip_syncer_weight = 0.5`
- `execution_providers = cuda`
- `video_memory_strategy = tolerant`
- `temp_frame_format = png`
- `output_video_encoder = h264_nvenc`
- `output_video_preset = fast`
- `output_video_quality = 90`

## File inventory

```
Changed files:
  facefusion/processors/modules/lip_syncer/core.py      (main logic, ~589 lines)
  facefusion/processors/modules/lip_syncer/choices.py    (added pure_motion_range)
  facefusion/processors/modules/lip_syncer/locales.py    (added locale strings)
  facefusion/uis/components/lip_syncer_options.py         (added slider + handler)
  facefusion/uis/components/preview.py                    (added preview trigger)
  facefusion/uis/types.py                                 (added component name)
  facefusion/config.py                                    (save current UI state to config)
  facefusion/uis/components/common_options.py             (SAVE AS DEFAULT button + status)
  facefusion/locales.py                                   (SAVE AS DEFAULT strings)
  facefusion.ini                                          (added lip_syncer_pure_motion key)

Unchanged but relevant:
  facefusion/processors/modules/lip_syncer/types.py       (LipSyncerInputs, LipSyncerModel, LipSyncerWeight)
  facefusion/processors/live_portrait.py                   (create_rotation, limit_expression, EXPRESSION_MIN/MAX)
  facefusion/processors/types.py                           (LivePortrait* type aliases)
  facefusion/processors/modules/face_editor/core.py        (reference implementation for LivePortrait forwards)
  facefusion/processors/modules/expression_restorer/core.py (another LivePortrait reference)
  facefusion/inference_manager.py                           (pool caching, context key logic)
  facefusion/download.py                                    (conditional_download_*, curl --create-dirs)

Reference:
  pure_motion_2.patch                                      (original patch this was ported from)
```
