#!/bin/bash

# Set the checkpoints directory
CheckpointsDir="models"

# Create necessary directories (baseline + optional GPEN-BFR)
mkdir -p models/musetalkV15 models/face-parse-bisent models/sd-vae models/whisper models/face_detection/weights models/gpen_bfr

# Set HuggingFace mirror endpoint
export HF_ENDPOINT=https://hf-mirror.com

# Download MuseTalk V1.5 weights (unet.pth)
python -m huggingface_hub.commands.huggingface_cli download TMElyralab/MuseTalk \
  --local-dir $CheckpointsDir \
  --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"

# Download SD VAE weights
python -m huggingface_hub.commands.huggingface_cli download stabilityai/sd-vae-ft-mse \
  --local-dir $CheckpointsDir/sd-vae \
  --include "config.json" "diffusion_pytorch_model.bin"

# Download Whisper weights
python -m huggingface_hub.commands.huggingface_cli download openai/whisper-tiny \
  --local-dir $CheckpointsDir/whisper \
  --include "config.json" "pytorch_model.bin" "preprocessor_config.json"

# Download Face Parse Bisent weights
gdown --id 154JgKpzCPW82qINcVieuPH3fZ2e0P812 -O $CheckpointsDir/face-parse-bisent/79999_iter.pth
curl -L https://download.pytorch.org/models/resnet18-5c106cde.pth \
  -o $CheckpointsDir/face-parse-bisent/resnet18-5c106cde.pth

# Download GPEN-BFR models (optional enhancement)
echo "📥 Downloading GPEN-BFR model..."
curl -L "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/gpen_bfr_256.onnx" \
  -o $CheckpointsDir/gpen_bfr/gpen_bfr_256.onnx

# Download YOLOv8 Face Detection ONNX model
echo "📥 Downloading YOLOv8 Face Detection model..."
curl -L "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/yoloface_8n.onnx" \
  -o $CheckpointsDir/face_detection/weights/yoloface_8n.onnx

echo "✅ All weights have been downloaded successfully!" 
