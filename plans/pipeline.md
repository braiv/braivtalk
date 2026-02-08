# BraivTalk CLI Inference Pipeline (Current)

This repo is intentionally **CLI-first**. The single supported inference entrypoint is:

- `python app/scripts/inference.py --inference_config configs/inference/test.yaml ...`
- (recommended) `./inference.sh v1.5` or `inference.bat v1.5`

The pipeline is designed to **never crash/freeze on cutaways** by passing through frames with **no detected face**.

---

## Pipeline overview (CPU vs GPU)

### Stages

1. **Model init** (GPU)
   - Load MuseTalk **VAE + UNet + PositionalEncoding**
   - Load **WhisperModel** (Transformers)
   - Init **FaceParsing** (runs on GPU when CUDA is available)
   - Optional: init **GPEN-BFR** face enhancer (ONNX Runtime)

2. **Frame extraction** (CPU)
   - FFmpeg extracts PNG frames from input video (or accept a single image / image directory)

3. **Audio feature extraction** (GPU)
   - Librosa loads audio → Transformers feature extractor builds mel features
   - Whisper encoder produces hidden states
   - Features are chunked per video frame (based on FPS)

4. **Face detection (+ landmarks)** (Hybrid: GPU/CPU)
   - Primary: **YOLOv8 face detector (ONNX)** returning bbox + 5 landmarks
   - Fallback: **SFD FaceAlignment** (bbox only)
   - Frames without a face get a `coord_placeholder` marker (passthrough)

5. **Face crop + VAE encode** (GPU)
   - For face frames: crop bbox → resize 256×256 → VAE encode latents
   - For non-face frames: store `None` latent (passthrough)

6. **Batch inference (UNet) + VAE decode** (GPU)
   - `datagen_enhanced()` builds batches with a mix of `process` vs `passthrough`
   - UNet predicts latents conditioned on audio
   - VAE decodes generated face patches (256×256)

7. **Optional face enhancement (GPEN-BFR)** (Hybrid: GPU/CPU)
   - ONNX Runtime enhancement pass on decoded 256×256 face patches

8. **Blend back into original frame** (Hybrid: CPU + FaceParsing GPU if available)
   - Resize generated patch to bbox size
   - Build a blending mask (optionally landmark-driven “surgical” mouth mask)
   - Refine with FaceParsing segmentation
   - Composite generated region into the original frame

9. **Encode video + mux audio** (Hybrid)
   - FFmpeg encodes frames to MP4
   - Tries GPU encoders (NVENC/AMF/QSV) then CPU fallback
   - FFmpeg muxes audio onto the encoded video

---

## Flow chart

```mermaid
flowchart TD
  A[Inputs: video + audio] --> B[FFmpeg: extract frames<br/>CPU]
  A --> C[Audio: librosa + Whisper encoder + chunking<br/>GPU]

  B --> D[Face detection per frame<br/>Hybrid: YOLOv8 ONNX (CUDA/CPU) or SFD fallback]
  D --> E{Face found?}

  E -- No --> F[Passthrough frame<br/>CPU]
  E -- Yes --> G[Crop + resize 256x256<br/>CPU]
  G --> H[VAE encode latents<br/>GPU]

  C --> I[datagen_enhanced batching<br/>CPU]
  H --> I

  I --> J[UNet inference (audio-conditioned)<br/>GPU]
  J --> K[VAE decode face patch<br/>GPU]

  K --> L{GPEN-BFR enabled?}
  L -- Yes --> M[GPEN-BFR enhance (ONNX)<br/>Hybrid]
  L -- No --> N[Skip]

  M --> O[Blend into original frame<br/>Hybrid: CPU + FaceParsing GPU]
  N --> O
  F --> P[Write output frames<br/>CPU]
  O --> P

  P --> Q[FFmpeg encode + mux audio<br/>Hybrid]
  Q --> R[Output MP4]
```

---

## Key files

- **Entrypoint**: `app/scripts/inference.py`
- **Face detection / landmarks**: `app/braivtalk/utils/preprocessing.py`
- **Audio feature extraction**: `app/braivtalk/utils/audio_processor.py`
- **Batching (passthrough-aware)**: `app/braivtalk/utils/utils.py` (`datagen_enhanced`)
- **Blending + mouth mask shapes**: `app/braivtalk/utils/blending.py`
- **Face parsing segmentation**: `app/braivtalk/utils/face_parsing/`
- **Face enhancement (optional)**: `app/braivtalk/enhancers/` (GPEN-BFR ONNX wrapper + presets)

