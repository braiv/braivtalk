"""
GPEN-BFR 256 Face Enhancer
=========================

Wrapper for GPEN-BFR 256 face enhancement using ONNX Runtime.
Designed for 256x256 face images (MuseTalk VAE decode output).
"""

# Suppress ONNX Runtime verbose warnings BEFORE any imports
import os
import warnings
import logging

os.environ["ORT_LOGGING_LEVEL"] = "4"  # Fatal errors only
os.environ["OMP_NUM_THREADS"] = "1"  # Reduce threading warnings
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("onnxruntime").setLevel(logging.ERROR)

import cv2
import numpy as np
from typing import List, Optional, Dict, Any

from .gpen_bfr_parameter_configs import (
    GPEN_BFR_CONFIGS,
    apply_preprocessing,
    apply_postprocessing,
    get_config_by_name,
)


_onnx_available = None


def _check_onnx_availability():
    """Check if ONNX runtime is available and return appropriate providers"""
    global _onnx_available

    if _onnx_available is not None:
        return _onnx_available

    try:
        import onnxruntime as ort

        available_providers = ort.get_available_providers()

        providers = []
        if "CUDAExecutionProvider" in available_providers:
            providers.append("CUDAExecutionProvider")
            print("CUDA provider available for GPEN-BFR")
        if "CPUExecutionProvider" in available_providers:
            providers.append("CPUExecutionProvider")

        _onnx_available = {"available": True, "onnxruntime": ort, "providers": providers}

        print(f"ONNX Runtime available with providers: {providers}")
        return _onnx_available

    except ImportError as e:
        print(f"ONNX Runtime not available: {e}")
        print("Install with: pip install onnxruntime-gpu")
        _onnx_available = {"available": False, "error": str(e)}
        return _onnx_available


class GPENBFREnhancer:
    def __init__(
        self,
        model_path: str = "models/gpen_bfr/gpen_bfr_256.onnx",
        device: str = "auto",
        config_name: str = "CONSERVATIVE",
        custom_config: Optional[Dict[str, Any]] = None,
    ):
        onnx_info = _check_onnx_availability()
        if not onnx_info["available"]:
            raise ImportError(f"ONNX Runtime not available: {onnx_info.get('error', 'Unknown error')}")

        self.ort = onnx_info["onnxruntime"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"GPEN-BFR model not found: {model_path}\n"
                f"Download with: python download_weights.bat (Windows) or ./download_weights.sh (Linux/macOS)"
            )

        self.model_path = model_path
        self.device = device

        if custom_config is not None:
            self.config = custom_config
        else:
            self.config = get_config_by_name(config_name) if GPEN_BFR_CONFIGS is not None else get_config_by_name("CONSERVATIVE")

        print(f"Using enhancement config: {self.config.get('name', config_name)}")

        self.providers = self._setup_providers(device, onnx_info["providers"])

        try:
            import sys
            from io import StringIO

            session_options = self.ort.SessionOptions()
            session_options.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.log_severity_level = 4
            session_options.enable_profiling = False

            old_stderr = sys.stderr
            sys.stderr = StringIO()

            try:
                self.session = self.ort.InferenceSession(model_path, sess_options=session_options, providers=self.providers)
            finally:
                sys.stderr = old_stderr

            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
            self.output_shape = self.session.get_outputs()[0].shape

            print("GPEN-BFR initialized successfully")
            print(f"   Model: {os.path.basename(model_path)}")
            print(f"   Providers: {self.session.get_providers()}")
            print(f"   Input shape: {self.input_shape}")
            print(f"   Output shape: {self.output_shape}")

        except Exception as e:
            raise RuntimeError(f"Failed to initialize GPEN-BFR session: {e}")

    def _setup_providers(self, device: str, available_providers: List[str]) -> List[str]:
        providers = []

        if device in ("auto", "cuda"):
            if "CUDAExecutionProvider" in available_providers:
                providers.append(
                    (
                        "CUDAExecutionProvider",
                        {
                            "device_id": 0,
                            "arena_extend_strategy": "kNextPowerOfTwo",
                            "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                            "cudnn_conv_algo_search": "EXHAUSTIVE",
                            "do_copy_in_default_stream": True,
                        },
                    )
                )

        if device in ("auto", "cpu"):
            if "CPUExecutionProvider" in available_providers:
                providers.append(
                    (
                        "CPUExecutionProvider",
                        {
                            "intra_op_num_threads": 4,
                            "inter_op_num_threads": 4,
                        },
                    )
                )

        if not providers:
            if "CUDAExecutionProvider" in available_providers:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                providers = ["CPUExecutionProvider"]

        return providers

    def preprocess_face(self, face_image: np.ndarray) -> np.ndarray:
        if face_image is None or face_image.size == 0:
            raise ValueError("Invalid input face image")

        if face_image.shape[:2] != (256, 256):
            face_image = cv2.resize(face_image, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        if len(face_image.shape) == 3 and face_image.shape[2] == 3:
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        face_tensor = face_image.astype(np.float32) / 127.5 - 1.0
        face_tensor = np.transpose(face_tensor, (2, 0, 1))
        face_tensor = np.expand_dims(face_tensor, axis=0)
        return face_tensor

    def postprocess_face(self, output_tensor: np.ndarray) -> np.ndarray:
        output_tensor = np.squeeze(output_tensor, axis=0)
        output_tensor = np.transpose(output_tensor, (1, 2, 0))
        output_tensor = (output_tensor + 1.0) * 127.5
        output_tensor = np.clip(output_tensor, 0, 255).astype(np.uint8)
        output_tensor = cv2.cvtColor(output_tensor, cv2.COLOR_RGB2BGR)
        return output_tensor

    def enhance_face(self, face_image: np.ndarray) -> np.ndarray:
        try:
            original_face = face_image.copy()
            if original_face.shape[:2] != (256, 256):
                original_face = cv2.resize(original_face, (256, 256), interpolation=cv2.INTER_LANCZOS4)

            preprocessed = apply_preprocessing(original_face, self.config) if GPEN_BFR_CONFIGS is not None else original_face

            input_tensor = self.preprocess_face(preprocessed)
            output_tensor = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
            enhanced_face = self.postprocess_face(output_tensor)

            if GPEN_BFR_CONFIGS is not None:
                enhanced_face = apply_postprocessing(original_face, enhanced_face, self.config)

            return enhanced_face
        except Exception as e:
            print(f"WARNING: GPEN-BFR enhancement failed for face: {e}")
            if face_image.shape[:2] != (256, 256):
                return cv2.resize(face_image, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            return face_image.copy()

    def enhance_batch(self, face_images: List[np.ndarray], show_progress: bool = True) -> List[np.ndarray]:
        enhanced_faces = []
        total_faces = len(face_images)

        for i, face in enumerate(face_images):
            if show_progress:
                print(f"Enhancing face {i+1}/{total_faces} with GPEN-BFR...")
            enhanced_faces.append(self.enhance_face(face))

        if show_progress:
            print(f"Enhanced {total_faces} faces with GPEN-BFR")

        return enhanced_faces

