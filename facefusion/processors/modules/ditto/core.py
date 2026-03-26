from argparse import ArgumentParser
from functools import lru_cache
from typing import Dict, List, Optional

import cv2
import numpy

import facefusion.jobs.job_manager
import facefusion.jobs.job_store
from facefusion import config, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, logger, state_manager, translator, video_manager
from facefusion.common_helper import get_first
from facefusion.face_analyser import scale_face
from facefusion.face_selector import select_faces
from facefusion.filesystem import filter_audio_paths, has_audio, in_directory, is_image, is_video, resolve_relative_path
from facefusion.processors.modules.ditto import choices as ditto_choices
from facefusion.processors.modules.ditto.locales import get as locale_get
from facefusion.processors.modules.ditto.types import DittoInputs
from facefusion.processors.types import ProcessorOutputs
from facefusion.program_helper import find_argument_group
from facefusion.types import ApplyStateItem, Args, InferencePool, ProcessMode
from facefusion.vision import read_static_image, read_static_video_frame

import threading

from ditto.runtime import DittoRunner

DITTO_RUNNER : Optional[DittoRunner] = None
DITTO_MODEL_DIR = resolve_relative_path('../.assets/models/ditto')
_PIPELINE_CACHE_KEY : Optional[str] = None
_PIPELINE_FAILED : bool = False
_PIPELINE_LOCK = threading.Lock()


def get_inference_pool() -> InferencePool:
	return {}


def clear_inference_pool() -> None:
	global DITTO_RUNNER, _PIPELINE_CACHE_KEY, _PIPELINE_FAILED

	if DITTO_RUNNER is not None:
		DITTO_RUNNER.teardown()
		DITTO_RUNNER = None
	_PIPELINE_CACHE_KEY = None
	_PIPELINE_FAILED = False


def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument('--ditto-source-mode', help = locale_get('help.source_mode'), default = config.get_str_value('processors', 'ditto_source_mode', 'video_native'), choices = ditto_choices.ditto_source_modes)
		group_processors.add_argument('--ditto-backend', help = locale_get('help.backend'), default = config.get_str_value('processors', 'ditto_backend', 'onnx'), choices = ditto_choices.ditto_backends)
		group_processors.add_argument('--ditto-render-mode', help = locale_get('help.render_mode'), default = config.get_str_value('processors', 'ditto_render_mode', 'native_putback'), choices = ditto_choices.ditto_render_modes)
		facefusion.jobs.job_store.register_step_keys([ 'ditto_source_mode', 'ditto_backend', 'ditto_render_mode' ])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('ditto_source_mode', args.get('ditto_source_mode'))
	apply_state_item('ditto_backend', args.get('ditto_backend'))
	apply_state_item('ditto_render_mode', args.get('ditto_render_mode'))


def pre_check() -> bool:
	return True


def pre_process(mode : ProcessMode) -> bool:
	if not has_audio(state_manager.get_item('source_paths')):
		logger.error(translator.get('choose_audio_source') + translator.get('exclamation_mark'), __name__)
		return False
	if mode in [ 'output', 'preview' ] and not is_image(state_manager.get_item('target_path')) and not is_video(state_manager.get_item('target_path')):
		logger.error(translator.get('choose_image_or_video_target') + translator.get('exclamation_mark'), __name__)
		return False
	if mode == 'output' and not in_directory(state_manager.get_item('output_path')):
		logger.error(translator.get('specify_image_or_video_output') + translator.get('exclamation_mark'), __name__)
		return False
	return True


def post_process() -> None:
	read_static_image.cache_clear()
	read_static_video_frame.cache_clear()
	video_manager.clear_video_pool()
	if state_manager.get_item('video_memory_strategy') in [ 'strict', 'moderate' ]:
		clear_inference_pool()
	if state_manager.get_item('video_memory_strategy') == 'strict':
		content_analyser.clear_inference_pool()
		face_classifier.clear_inference_pool()
		face_detector.clear_inference_pool()
		face_landmarker.clear_inference_pool()
		face_masker.clear_inference_pool()
		face_recognizer.clear_inference_pool()


def _build_cache_key() -> str:
	source_audio_path = get_first(filter_audio_paths(state_manager.get_item('source_paths'))) or ''
	target_path = state_manager.get_item('target_path') or ''
	return ':'.join([source_audio_path, target_path,
					 str(state_manager.get_item('ditto_source_mode')),
					 str(state_manager.get_item('ditto_backend'))])


def _load_target_frames(target_path : str, max_frames : int = 500) -> List:
	"""Load RGB frames from a video or image for Ditto avatar registration."""
	if is_image(target_path):
		img = cv2.imread(target_path)
		if img is None:
			return []
		return [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)]

	cap = cv2.VideoCapture(target_path)
	frames = []
	while cap.isOpened() and len(frames) < max_frames:
		ret, frame = cap.read()
		if not ret:
			break
		frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
	cap.release()
	return frames


def _ensure_pipeline_prepared() -> bool:
	"""Ensure the Ditto pipeline is prepared with current settings. Returns True if ready.

	Thread-safe: only one thread can prepare the pipeline at a time.
	Once preparation fails, further calls return False immediately until clear_inference_pool().
	"""
	global DITTO_RUNNER, _PIPELINE_CACHE_KEY, _PIPELINE_FAILED

	# Fast path: already prepared
	cache_key = _build_cache_key()
	if DITTO_RUNNER is not None and DITTO_RUNNER.is_prepared and _PIPELINE_CACHE_KEY == cache_key:
		return True

	# Fast path: already failed for this session
	if _PIPELINE_FAILED:
		return False

	with _PIPELINE_LOCK:
		# Re-check under lock (another thread may have prepared while we waited)
		if DITTO_RUNNER is not None and DITTO_RUNNER.is_prepared and _PIPELINE_CACHE_KEY == cache_key:
			return True
		if _PIPELINE_FAILED:
			return False

		try:
			return _prepare_pipeline_locked(cache_key)
		except Exception as e:
			logger.error(f'Ditto pipeline preparation failed: {e}', __name__)
			import traceback
			traceback.print_exc()
			_PIPELINE_FAILED = True
			return False


def _prepare_pipeline_locked(cache_key : str) -> bool:
	"""Runs inside _PIPELINE_LOCK. Performs the full preparation sequence."""
	global DITTO_RUNNER, _PIPELINE_CACHE_KEY, _PIPELINE_FAILED

	source_audio_path = get_first(filter_audio_paths(state_manager.get_item('source_paths')))
	target_path = state_manager.get_item('target_path')
	if not source_audio_path or not target_path:
		_PIPELINE_FAILED = True
		return False

	if DITTO_RUNNER is None:
		DITTO_RUNNER = DittoRunner()

	backend = state_manager.get_item('ditto_backend')
	source_mode = state_manager.get_item('ditto_source_mode')
	render_mode = state_manager.get_item('ditto_render_mode')
	DITTO_RUNNER.setup(model_dir=DITTO_MODEL_DIR, backend=backend,
					   source_mode=source_mode, render_mode=render_mode)

	if not DITTO_RUNNER.is_ready:
		logger.error('Ditto backend not available', __name__)
		_PIPELINE_FAILED = True
		return False

	if not DITTO_RUNNER.ensure_models():
		logger.error('Failed to download Ditto models', __name__)
		_PIPELINE_FAILED = True
		return False

	print('[ditto] Loading target video frames for avatar registration…')
	img_rgb_list = _load_target_frames(target_path)
	if not img_rgb_list:
		logger.error('Failed to load target frames', __name__)
		_PIPELINE_FAILED = True
		return False

	try:
		import librosa
		audio_16k, _ = librosa.core.load(source_audio_path, sr=16000)
	except ImportError:
		logger.error('librosa is required for Ditto. Install with: pip install librosa', __name__)
		_PIPELINE_FAILED = True
		return False
	except Exception as e:
		logger.error(f'Failed to load audio: {e}', __name__)
		_PIPELINE_FAILED = True
		return False

	DITTO_RUNNER.prepare(img_rgb_list, audio_16k, sampling_timesteps=50)
	_PIPELINE_CACHE_KEY = cache_key
	return True


def process_frame(inputs : DittoInputs) -> ProcessorOutputs:
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')
	frame_number = inputs.get('frame_number', 0)

	if not _ensure_pipeline_prepared():
		return temp_vision_frame, temp_vision_mask

	result_rgb = DITTO_RUNNER.process_frame(frame_number)
	if result_rgb is None:
		return temp_vision_frame, temp_vision_mask

	result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

	# Match output size to temp_vision_frame if they differ
	if result_bgr.shape[:2] != temp_vision_frame.shape[:2]:
		result_bgr = cv2.resize(result_bgr, (temp_vision_frame.shape[1], temp_vision_frame.shape[0]),
								interpolation=cv2.INTER_LINEAR)

	# Preserve alpha channel if present
	if temp_vision_frame.shape[2] == 4 and result_bgr.shape[2] == 3:
		result_bgra = numpy.dstack([result_bgr, temp_vision_frame[:, :, 3]])
		return result_bgra, temp_vision_mask

	return result_bgr, temp_vision_mask
