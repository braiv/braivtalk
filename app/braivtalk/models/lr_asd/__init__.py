"""
LR-ASD: Lightweight and Robust Network for Active Speaker Detection.

Vendored model architecture from https://github.com/Junhua-Liao/LR-ASD
Paper: IJCV 2025 / CVPR 2023
"""

from .detector import ActiveSpeakerDetector

__all__ = ["ActiveSpeakerDetector"]
