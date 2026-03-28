"""
Ditto runtime boundary – the narrow API consumed by the FaceFusion processor shell.

Wraps ditto.pipeline.DittoPipeline and handles:
  - Model download from HuggingFace
  - Pipeline lifecycle (load → prepare → render per frame → teardown)
  - Capability probing for available backends
"""

import os
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from numpy.typing import NDArray

from ditto.pipeline import (
    DittoPipeline,
    REQUIRED_ONNX_MODELS,
    check_models,
    download_models,
)

DittoSourceMode = Literal['video_native', 'image_anchor']
DittoBackend = Literal['onnx', 'trt', 'pytorch']
DittoRenderMode = Literal['native_putback', 'facefusion_composite']

HUGGINGFACE_REPO = 'digital-avatar/ditto-talkinghead'
HUGGINGFACE_BASE = f'https://huggingface.co/{HUGGINGFACE_REPO}/resolve/main'


class DittoCapabilities:
	def __init__(self) -> None:
		self.has_onnx = False
		self.has_trt = False
		self.has_pytorch = False
		self.available_backends : List[DittoBackend] = []


def probe_capabilities() -> DittoCapabilities:
	caps = DittoCapabilities()
	try:
		import onnxruntime  # noqa: F401
		caps.has_onnx = True
		caps.available_backends.append('onnx')
	except Exception:
		pass
	try:
		import tensorrt  # noqa: F401
		caps.has_trt = True
		caps.available_backends.append('trt')
	except Exception:
		pass
	try:
		import torch
		if torch.cuda.is_available():
			caps.has_pytorch = True
			caps.available_backends.append('pytorch')
	except Exception:
		pass
	return caps


def select_best_backend(caps : DittoCapabilities) -> Optional[DittoBackend]:
	for preferred in ['onnx', 'trt', 'pytorch']:
		if preferred in caps.available_backends:
			return preferred
	return None


class DittoRunner:
	"""Narrow adapter API consumed by the FaceFusion processor shell.

	Lifecycle:
	  1. setup()        – probe capabilities, select backend
	  2. ensure_models() – download ONNX models if missing
	  3. prepare()       – load models, register avatar, pre-compute motion
	  4. process_frame() – render a single frame using pre-computed motion
	  5. teardown()      – release memory
	"""

	def __init__(self) -> None:
		self._ready = False
		self._prepared = False
		self._backend : Optional[DittoBackend] = None
		self._source_mode : DittoSourceMode = 'video_native'
		self._render_mode : DittoRenderMode = 'native_putback'
		self._capabilities : Optional[DittoCapabilities] = None
		self._pipeline : Optional[DittoPipeline] = None
		self._model_dir : str = ''
		self._num_output_frames : int = 0

	def setup(self, model_dir : str, backend : Optional[DittoBackend] = None,
			  source_mode : DittoSourceMode = 'video_native',
			  render_mode : DittoRenderMode = 'native_putback') -> bool:
		self._capabilities = probe_capabilities()
		self._backend = backend or select_best_backend(self._capabilities)
		self._source_mode = source_mode
		self._render_mode = render_mode
		self._model_dir = model_dir
		self._ready = self._backend is not None
		return self._ready

	@property
	def is_ready(self) -> bool:
		return self._ready

	@property
	def is_prepared(self) -> bool:
		return self._prepared

	@property
	def backend(self) -> Optional[DittoBackend]:
		return self._backend

	@property
	def capabilities(self) -> Optional[DittoCapabilities]:
		return self._capabilities

	@property
	def num_output_frames(self) -> int:
		return self._num_output_frames

	def get_expected_face_hint(self, frame_number : int) -> Optional[Dict[str, NDArray]]:
		if self._pipeline is None or not self._prepared:
			return None
		return self._pipeline.get_expected_face_hint(frame_number)

	def ensure_models(self) -> bool:
		"""Download models if they don't exist. Returns True if all present."""
		if check_models(self._model_dir):
			return True
		print(f'[ditto] Models not found in {self._model_dir}, downloading...')
		return download_models(self._model_dir)

	def prepare(self, img_rgb_list : List[NDArray], audio_16k : NDArray,
				source_frame_numbers : Optional[List[int]] = None,
				source_total_frames : Optional[int] = None,
				crop_frames : Optional[List[Optional[NDArray]]] = None,
				affine_matrices : Optional[List[Optional[NDArray]]] = None,
				bbox_hints : Optional[List[Optional[NDArray]]] = None,
				landmark_hints : Optional[List[Optional[NDArray]]] = None,
				sampling_timesteps : int = 50, emo : int = 4) -> int:
		"""Load models and pre-compute all motion from source + audio.

		Returns the number of output frames.
		"""
		if not self._ready:
			raise RuntimeError('DittoRunner not set up')

		if not self.ensure_models():
			raise RuntimeError('Ditto models not available')

		self._pipeline = DittoPipeline(self._model_dir)
		self._pipeline.load_models()
		self._num_output_frames = self._pipeline.prepare(
			img_rgb_list, audio_16k,
			source_frame_numbers=source_frame_numbers,
			source_total_frames=source_total_frames,
			crop_frames=crop_frames,
			affine_matrices=affine_matrices,
			bbox_hints=bbox_hints,
			landmark_hints=landmark_hints,
			sampling_timesteps=sampling_timesteps, emo=emo,
		)
		self._prepared = True
		return self._num_output_frames

	def process_frame(self, frame_number : int = 0,
					current_frame_rgb : Optional[NDArray] = None,
					current_bbox : Optional[NDArray] = None,
					current_landmarks : Optional[NDArray] = None) -> Optional[NDArray]:
		"""Render a single output frame using pre-computed motion.

		Returns composited RGB frame or None.
		"""
		if not self._prepared or self._pipeline is None:
			return None
		return self._pipeline.render_frame(
			frame_number,
			current_frame_rgb=current_frame_rgb,
			current_bbox=current_bbox,
			current_landmarks=current_landmarks
		)

	def process_frame_data(self, frame_number : int = 0,
						current_frame_rgb : Optional[NDArray] = None,
						current_crop_rgb : Optional[NDArray] = None,
						current_affine_matrix : Optional[NDArray] = None,
						current_bbox : Optional[NDArray] = None,
						current_landmarks : Optional[NDArray] = None) -> Optional[Dict[str, NDArray]]:
		"""Render a single frame and return crop-space data."""
		if not self._prepared or self._pipeline is None:
			return None
		return self._pipeline.render_frame_data(
			frame_number,
			current_frame_rgb=current_frame_rgb,
			current_crop_rgb=current_crop_rgb,
			current_affine_matrix=current_affine_matrix,
			current_bbox=current_bbox,
			current_landmarks=current_landmarks
		)

	def teardown(self) -> None:
		if self._pipeline is not None:
			self._pipeline.teardown()
			self._pipeline = None
		self._ready = False
		self._prepared = False
		self._backend = None
		self._capabilities = None
		self._num_output_frames = 0
