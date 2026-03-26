# Ditto Runtime (Experimental)

Repo-local Ditto talking-head runtime boundary.

This directory houses the experimental Ditto inference runner that is consumed
by the FaceFusion processor adapter at `facefusion/processors/modules/ditto/`.

## Architecture

- `runtime.py` — Narrow adapter API: capability probe, model discovery,
  backend selection, and frame-level inference entry point.
- The FaceFusion processor shell in `facefusion/processors/modules/ditto/core.py`
  delegates to `DittoRunner` for all Ditto-specific work.

## Status

Phase 1 stub. The runner validates wiring and backend availability but does
not yet perform real Ditto inference. Real model loading and inference will
be added in subsequent phases after the quality gate confirms Ditto is worth
deeper integration.

## Models

Ditto ONNX checkpoints are hosted at:
https://huggingface.co/digital-avatar/ditto-talkinghead

The runner knows how to enumerate model URLs for ONNX, TRT, and PyTorch
backends. Actual download integration uses FaceFusion's existing download
infrastructure.
