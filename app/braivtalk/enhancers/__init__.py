"""
Enhancers (optional post-processing) for BraivTalk.

Currently includes:
- GPEN-BFR 256 ONNX enhancer
"""

__all__ = ["GPENBFREnhancer"]

try:
    from .gpen_bfr_enhancer import GPENBFREnhancer
except Exception:
    # Optional dependency (onnxruntime) might not be installed in all environments
    __all__ = []

