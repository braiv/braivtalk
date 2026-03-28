from argparse import ArgumentParser
from functools import lru_cache
from typing import Any, Dict, List, Optional

import cv2
import numpy

import facefusion.jobs.job_manager
import facefusion.jobs.job_store
from facefusion import config, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, logger, state_manager, translator, video_manager
from facefusion.common_helper import get_first
from facefusion.face_analyser import get_many_faces, scale_face
from facefusion.face_helper import estimate_matrix_by_face_landmark_5
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
_PROCESS_LOCK = threading.Lock()


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
		group_processors.add_argument('--ditto-registration-crop-mode', help = locale_get('help.registration_crop_mode'), default = config.get_str_value('processors', 'ditto_registration_crop_mode', 'ditto_native'), choices = ditto_choices.ditto_crop_prep_modes)
		group_processors.add_argument('--ditto-live-crop-mode', help = locale_get('help.live_crop_mode'), default = config.get_str_value('processors', 'ditto_live_crop_mode', 'ditto_native'), choices = ditto_choices.ditto_crop_prep_modes)
		group_processors.add_argument('--ditto-composite-geometry-mode', help = locale_get('help.composite_geometry_mode'), default = config.get_str_value('processors', 'ditto_composite_geometry_mode', 'ditto_transform'), choices = ditto_choices.ditto_composite_geometry_modes)
		facefusion.jobs.job_store.register_step_keys([ 'ditto_source_mode', 'ditto_backend', 'ditto_render_mode', 'ditto_registration_crop_mode', 'ditto_live_crop_mode', 'ditto_composite_geometry_mode' ])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('ditto_source_mode', args.get('ditto_source_mode'))
	apply_state_item('ditto_backend', args.get('ditto_backend'))
	apply_state_item('ditto_render_mode', args.get('ditto_render_mode'))
	apply_state_item('ditto_registration_crop_mode', args.get('ditto_registration_crop_mode'))
	apply_state_item('ditto_live_crop_mode', args.get('ditto_live_crop_mode'))
	apply_state_item('ditto_composite_geometry_mode', args.get('ditto_composite_geometry_mode'))


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


def _load_target_frames(target_path : str, max_frames : Optional[int] = None) -> Dict[str, Any]:
	"""Load RGB frames across the full target timeline for avatar registration."""
	if is_image(target_path):
		img = cv2.imread(target_path)
		if img is None:
			return {'frames': [], 'frame_numbers': [], 'total_frames': 0}
		return {
			'frames': [cv2.cvtColor(img, cv2.COLOR_BGR2RGB)],
			'frame_numbers': [0],
			'total_frames': 1
		}

	cap = cv2.VideoCapture(target_path)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
	frames = []
	frame_numbers = []
	if max_frames is not None and total_frames > 0 and total_frames > max_frames:
		sampled_numbers = numpy.unique(
			numpy.linspace(0, total_frames - 1, num=max_frames, dtype=numpy.int32)
		).tolist()
	else:
		sampled_numbers = None
	sampled_set = set(sampled_numbers or [])
	frame_index = 0
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break
		if sampled_numbers is None or frame_index in sampled_set:
			frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
			frame_numbers.append(frame_index)
		frame_index += 1
	cap.release()
	if total_frames <= 0:
		total_frames = frame_index
	if sampled_numbers is not None and len(frame_numbers) != len(sampled_numbers):
		logger.warning(f'Ditto loaded {len(frame_numbers)} of {len(sampled_numbers)} sampled frames', __name__)
	return {
		'frames': frames,
		'frame_numbers': frame_numbers,
		'total_frames': total_frames
	}


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

	print('[ditto] Loading target video frames for avatar registration...')
	target_frames = _load_target_frames(target_path)
	img_rgb_list = target_frames.get('frames', [])
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

	if is_image(target_path):
		reference_vision_frame = read_static_image(target_path)
	else:
		reference_vision_frame = read_static_video_frame(target_path, state_manager.get_item('reference_frame_number'))

	use_facefusion_registration_crop = get_ditto_registration_crop_mode() == 'facefusion_crop'
	crop_frames = []
	affine_matrices = []
	bbox_hints = []
	landmark_hints = []
	for frame_number, rgb_frame in zip(target_frames.get('frame_numbers', []), img_rgb_list):
		target_vision_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
		target_faces = select_faces(reference_vision_frame, target_vision_frame) if reference_vision_frame is not None else []
		if not target_faces:
			if use_facefusion_registration_crop:
				crop_frames.append(None)
				affine_matrices.append(None)
			bbox_hints.append(None)
			landmark_hints.append(None)
			continue
		target_face = target_faces[0]
		if use_facefusion_registration_crop:
			crop_rgb, affine_matrix = _create_facefusion_crop(target_vision_frame, target_face, frame_number, stabilize = False)
			crop_frames.append(crop_rgb)
			affine_matrices.append(affine_matrix)
		bbox_hints.append(target_face.bounding_box.astype(numpy.float32))
		current_landmarks = target_face.landmark_set.get('68')
		if current_landmarks is None:
			current_landmarks = target_face.landmark_set.get('5/68')
		landmark_hints.append(current_landmarks.astype(numpy.float32) if current_landmarks is not None else None)

	DITTO_RUNNER.prepare(
		img_rgb_list,
		audio_16k,
		source_frame_numbers=target_frames.get('frame_numbers'),
		source_total_frames=target_frames.get('total_frames'),
		crop_frames=crop_frames if use_facefusion_registration_crop else None,
		affine_matrices=affine_matrices if use_facefusion_registration_crop else None,
		bbox_hints=bbox_hints,
		landmark_hints=landmark_hints,
		sampling_timesteps=50
	)
	_PIPELINE_CACHE_KEY = cache_key
	return True


def _bbox_center_size(bounding_box) -> tuple[numpy.ndarray, float]:
	x1, y1, x2, y2 = [ float(v) for v in bounding_box[:4] ]
	center = numpy.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype = numpy.float32)
	size = max(x2 - x1, y2 - y1)
	return center, size


def _get_ditto_face_rejection_reason(target_face, temp_vision_frame, frame_number : int) -> Optional[str]:
	"""Reject clearly unsafe faces before Ditto renders a full face patch.

	This intentionally keeps FaceFusion's normal candidate generation
	(`select_faces`) but adds a stricter acceptance step because Ditto's
	full-face putback is much less forgiving than the default lip-sync masks.
	"""
	detector_score = float(target_face.score_set.get('detector') or 0.0)
	landmarker_score = float(target_face.score_set.get('landmarker') or 0.0)
	min_detector_score = max(float(state_manager.get_item('face_detector_score') or 0.0), 0.70)
	if detector_score < min_detector_score:
		return f'detector<{min_detector_score:.2f} ({detector_score:.2f})'
	min_landmarker_score = max(float(state_manager.get_item('face_landmarker_score') or 0.0), 0.65)
	if landmarker_score > 0 and landmarker_score < min_landmarker_score:
		return f'landmarker<{min_landmarker_score:.2f} ({landmarker_score:.2f})'

	frame_h, frame_w = temp_vision_frame.shape[:2]
	_, bbox_size = _bbox_center_size(target_face.bounding_box)
	bbox_area = (target_face.bounding_box[2] - target_face.bounding_box[0]) * (target_face.bounding_box[3] - target_face.bounding_box[1])
	frame_area = max(frame_h * frame_w, 1)
	area_ratio = float(bbox_area) / float(frame_area)
	if area_ratio < 0.005 or area_ratio > 0.20:
		return f'area_ratio={area_ratio:.4f}'

	if DITTO_RUNNER is not None and DITTO_RUNNER.is_prepared:
		expected_hint = DITTO_RUNNER.get_expected_face_hint(frame_number)
		if expected_hint:
			expected_center = numpy.asarray(expected_hint.get('center'), dtype = numpy.float32)
			expected_size = float(expected_hint.get('size') or 0.0)
			current_center, current_size = _bbox_center_size(target_face.bounding_box)
			if expected_size > 1e-3 and current_size > 1e-3:
				center_penalty = float(numpy.linalg.norm(current_center - expected_center)) / max(expected_size, current_size, 1.0)
				size_ratio = current_size / expected_size
				if center_penalty > 1.4 or size_ratio < 0.35 or size_ratio > 2.5:
					return f'hint_mismatch d={center_penalty:.2f} size={size_ratio:.2f}'

	return None


def _draw_face_box(debug_frame, target_face, color, label : str) -> None:
	x1, y1, x2, y2 = [ int(v) for v in target_face.bounding_box[:4] ]
	cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
	cv2.putText(debug_frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _draw_landmarks(debug_frame, landmarks, color) -> None:
	if landmarks is None:
		return
	for point in landmarks.astype(numpy.int32):
		cv2.circle(debug_frame, tuple(point[:2]), 1, color, -1)


def _draw_expected_hint(debug_frame, frame_number : int) -> None:
	return _get_expected_hint_debug_info(frame_number, debug_frame)


def _get_expected_hint_debug_info(frame_number : int, debug_frame = None) -> Optional[Dict[str, float]]:
	if DITTO_RUNNER is None or not DITTO_RUNNER.is_prepared:
		return None
	expected_hint = DITTO_RUNNER.get_expected_face_hint(frame_number)
	if not expected_hint:
		return None
	center = numpy.asarray(expected_hint.get('center'), dtype = numpy.float32)
	size = float(expected_hint.get('size') or 0.0)
	if center.size < 2 or size <= 0:
		return None
	cx, cy = int(center[0]), int(center[1])
	half = int(size / 2)
	if debug_frame is not None:
		cv2.rectangle(debug_frame, (cx - half, cy - half), (cx + half, cy + half), (255, 0, 255), 2)
		cv2.circle(debug_frame, (cx, cy), 4, (255, 0, 255), -1)
		cv2.putText(debug_frame, 'expected', (cx - half, max(20, cy - half - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2, cv2.LINE_AA)
	return {
		'cx': float(center[0]),
		'cy': float(center[1]),
		'size': size,
		'source_index': float(expected_hint.get('source_index') or 0),
		'source_frame_number': float(expected_hint.get('source_frame_number') or 0),
		'source_total_frames': float(expected_hint.get('source_total_frames') or 0)
	}


def _create_debug_frame(target_vision_frame, frame_number : int, raw_faces = None, selected_faces = None, selected_face = None, warning_reason : Optional[str] = None, accepted : bool = True):
	debug_frame = target_vision_frame[:, :, :3].copy()
	raw_faces = raw_faces or []
	selected_faces = selected_faces or []
	status_lines = [
		f'frame={frame_number}',
		f'raw_faces={len(raw_faces)}',
		f'selected_faces={len(selected_faces)}'
	]

	expected_hint_info = _get_expected_hint_debug_info(frame_number, debug_frame)
	if expected_hint_info:
		status_lines.append(
			f'expected_sample={int(expected_hint_info["source_frame_number"])}/{max(int(expected_hint_info["source_total_frames"]) - 1, 0)} idx={int(expected_hint_info["source_index"])}'
		)
		status_lines.append(
			f'expected_center=({int(expected_hint_info["cx"])},{int(expected_hint_info["cy"])}) size={expected_hint_info["size"]:.1f}'
		)
	else:
		status_lines.append('expected_sample=None')

	for index, face in enumerate(raw_faces):
		_draw_face_box(debug_frame, face, (0, 255, 255), f'raw[{index}] d={float(face.score_set.get("detector") or 0.0):.2f}')

	for index, face in enumerate(selected_faces):
		_draw_face_box(debug_frame, face, (255, 255, 0), f'selected[{index}]')

	if selected_face is not None:
		label = 'accepted'
		color = (0, 200, 0)
		if not accepted:
			label = f'rejected: {warning_reason or "unknown"}'
			color = (0, 0, 255)
		elif warning_reason:
			label = f'accepted: {warning_reason}'
			color = (0, 165, 255)
		_draw_face_box(debug_frame, selected_face, color, label)
		_draw_landmarks(debug_frame, selected_face.landmark_set.get('68'), color)
		detector_score = float(selected_face.score_set.get('detector') or 0.0)
		landmarker_score = float(selected_face.score_set.get('landmarker') or 0.0)
		status_lines.append(f'det={detector_score:.2f} lmk={landmarker_score:.2f}')
	else:
		status_lines.append('selected_face=None')

	if warning_reason:
		status_lines.append(f'hint_warning={warning_reason}')
	elif selected_face is not None:
		status_lines.append('selection_source=facefusion')
	else:
		status_lines.append('selection=none')

	for index, line in enumerate(status_lines):
		cv2.putText(debug_frame, line, (12, 28 + index * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
		cv2.putText(debug_frame, line, (12, 28 + index * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
	return debug_frame


def _paste_ditto_with_facefusion(temp_vision_frame, target_face, render_rgb, affine_matrix, debug_mask : bool = False):
	from facefusion.face_helper import paste_back
	from facefusion.processors.modules.lip_syncer.core import create_live_portrait_mask, create_occlusion_masks, visualize_mask

	render_bgr = cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)
	crop_masks = create_occlusion_masks(render_bgr)
	crop_masks.append(create_live_portrait_mask(render_bgr, target_face, affine_matrix))
	crop_mask = numpy.minimum.reduce(crop_masks)

	if debug_mask:
		return visualize_mask(temp_vision_frame, crop_mask, affine_matrix)
	return paste_back(temp_vision_frame, render_bgr, crop_mask, affine_matrix)


def _create_facefusion_crop(temp_vision_frame, target_face, frame_number : int, stabilize : bool = True):
	from facefusion.processors.modules.lip_syncer.core import stabilize_crop_face_landmark_5

	face_landmark_5 = target_face.landmark_set.get('5/68').astype(numpy.float32)
	if stabilize:
		stabilized_landmarks = stabilize_crop_face_landmark_5(target_face, temp_vision_frame, temp_vision_frame, frame_number)
		if stabilized_landmarks is not None and getattr(stabilized_landmarks, 'shape', (0, 0))[0] == 5:
			face_landmark_5 = stabilized_landmarks.astype(numpy.float32)
	affine_matrix = estimate_matrix_by_face_landmark_5(face_landmark_5, 'ffhq_512', (512, 512))
	crop_bgr = cv2.warpAffine(temp_vision_frame, affine_matrix, (512, 512), borderMode = cv2.BORDER_REPLICATE, flags = cv2.INTER_AREA)
	crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
	return crop_rgb, affine_matrix.astype(numpy.float32)


def get_ditto_registration_crop_mode() -> str:
	mode = state_manager.get_item('ditto_registration_crop_mode')
	if mode in ditto_choices.ditto_crop_prep_modes:
		return mode
	return 'ditto_native'


def get_ditto_live_crop_mode() -> str:
	mode = state_manager.get_item('ditto_live_crop_mode')
	if mode in ditto_choices.ditto_crop_prep_modes:
		return mode
	return 'ditto_native'


def get_ditto_composite_geometry_mode() -> str:
	mode = state_manager.get_item('ditto_composite_geometry_mode')
	if mode in ditto_choices.ditto_composite_geometry_modes:
		return mode
	return 'ditto_transform'


def _matrix_2x3_to_3x3(matrix) -> numpy.ndarray:
	return numpy.vstack([ matrix.astype(numpy.float32), numpy.array([ 0, 0, 1 ], dtype = numpy.float32) ])


def _warp_render_to_facefusion_space(render_rgb, ditto_affine_matrix, facefusion_affine_matrix):
	ditto_matrix = _matrix_2x3_to_3x3(ditto_affine_matrix)
	facefusion_matrix = _matrix_2x3_to_3x3(facefusion_affine_matrix)
	ditto_to_facefusion = facefusion_matrix @ numpy.linalg.inv(ditto_matrix)
	return cv2.warpAffine(render_rgb, ditto_to_facefusion[:2, :], (512, 512), flags = cv2.INTER_LINEAR, borderMode = cv2.BORDER_REPLICATE)


def _stabilize_ditto_landmarks(target_face, target_vision_frame, temp_vision_frame, frame_number : int):
	from facefusion.processors.modules.lip_syncer.core import get_adjacent_target_face

	current_landmarks = target_face.landmark_set.get('68')
	if current_landmarks is None:
		return target_face.landmark_set.get('5/68')

	crop_stabilization = state_manager.get_item('lip_syncer_crop_stabilization') or 0.0
	if crop_stabilization <= 0:
		return current_landmarks

	neighbor_landmarks = []
	for frame_offset in [ -1, 1 ]:
		neighbor_face = get_adjacent_target_face(target_face, target_vision_frame, temp_vision_frame, frame_number, frame_offset)
		if neighbor_face is not None:
			neighbor_landmark_68 = neighbor_face.landmark_set.get('68')
			if neighbor_landmark_68 is not None and neighbor_landmark_68.shape == current_landmarks.shape:
				neighbor_landmarks.append(neighbor_landmark_68.astype(numpy.float32))

	if not neighbor_landmarks:
		return current_landmarks

	temporal_landmarks = current_landmarks.astype(numpy.float32) * 0.5
	neighbor_weight = 0.5 / len(neighbor_landmarks)
	for neighbor_landmark_68 in neighbor_landmarks:
		temporal_landmarks += neighbor_landmark_68 * neighbor_weight

	return current_landmarks.astype(numpy.float32) * (1 - crop_stabilization) + temporal_landmarks * crop_stabilization


def process_frame(inputs : DittoInputs) -> ProcessorOutputs:
	reference_vision_frame = inputs.get('reference_vision_frame')
	target_vision_frame = inputs.get('target_vision_frame')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')
	frame_number = inputs.get('frame_number', 0)
	debug_mask = inputs.get('debug_mask', False)

	if not _ensure_pipeline_prepared():
		if debug_mask:
			return _create_debug_frame(target_vision_frame, frame_number), temp_vision_mask
		return temp_vision_frame, temp_vision_mask

	raw_faces = get_many_faces([ target_vision_frame ]) if debug_mask else []
	target_faces = select_faces(reference_vision_frame, target_vision_frame)
	if not target_faces:
		if debug_mask:
			return _create_debug_frame(target_vision_frame, frame_number, raw_faces = raw_faces, selected_faces = target_faces), temp_vision_mask
		return temp_vision_frame, temp_vision_mask

	selected_face = target_faces[0]
	target_face = scale_face(selected_face, target_vision_frame, temp_vision_frame)
	warning_reason = _get_ditto_face_rejection_reason(target_face, temp_vision_frame, frame_number)
	if debug_mask:
		return _create_debug_frame(
			target_vision_frame,
			frame_number,
			raw_faces = raw_faces,
			selected_faces = target_faces,
			selected_face = selected_face,
			warning_reason = warning_reason,
			accepted = True
		), temp_vision_mask
	current_frame_rgb = cv2.cvtColor(target_vision_frame[:, :, :3], cv2.COLOR_BGR2RGB)
	current_bbox = target_face.bounding_box.astype(numpy.float32)
	current_landmarks = _stabilize_ditto_landmarks(target_face, target_vision_frame, temp_vision_frame[:, :, :3], frame_number)
	if current_landmarks is not None:
		current_landmarks = current_landmarks.astype(numpy.float32)
	affine_matrix = None
	current_crop_rgb = None
	facefusion_crop_rgb = None
	facefusion_affine_matrix = None
	if get_ditto_live_crop_mode() == 'facefusion_crop':
		current_crop_rgb, affine_matrix = _create_facefusion_crop(temp_vision_frame[:, :, :3], target_face, frame_number, stabilize = True)
		facefusion_crop_rgb = current_crop_rgb
		facefusion_affine_matrix = affine_matrix
	elif get_ditto_composite_geometry_mode() in ( 'facefusion_affine', 'warp_to_facefusion' ):
		facefusion_crop_rgb, facefusion_affine_matrix = _create_facefusion_crop(temp_vision_frame[:, :, :3], target_face, frame_number, stabilize = True)

	# Ditto's render path uses heavyweight GPU ops and live face analysis.
	# FaceFusion processes video frames in parallel, so guard Ditto to avoid
	# overlapping allocations and out-of-order state issues.
	with _PROCESS_LOCK:
		render_data = DITTO_RUNNER.process_frame_data(
			frame_number,
			current_frame_rgb=current_frame_rgb,
			current_crop_rgb=current_crop_rgb,
			current_affine_matrix=affine_matrix,
			current_bbox=current_bbox,
			current_landmarks=current_landmarks
		)
	if render_data is None:
		return temp_vision_frame, temp_vision_mask
	composite_geometry_mode = get_ditto_composite_geometry_mode()
	render_rgb = render_data.get('render_img')
	if composite_geometry_mode == 'ditto_transform':
		if affine_matrix is None:
			affine_matrix = render_data.get('M_o2c')[:2, :]
	elif composite_geometry_mode == 'facefusion_affine':
		affine_matrix = facefusion_affine_matrix if facefusion_affine_matrix is not None else render_data.get('M_o2c')[:2, :]
	elif composite_geometry_mode == 'warp_to_facefusion':
		if facefusion_affine_matrix is not None:
			render_rgb = _warp_render_to_facefusion_space(render_rgb, render_data.get('M_o2c')[:2, :], facefusion_affine_matrix)
			affine_matrix = facefusion_affine_matrix
		else:
			affine_matrix = render_data.get('M_o2c')[:2, :]
	else:
		affine_matrix = render_data.get('M_o2c')[:2, :]
	result_bgr = _paste_ditto_with_facefusion(
		temp_vision_frame[:, :, :3],
		target_face,
		render_rgb,
		affine_matrix,
		debug_mask
	)

	# Match output size to temp_vision_frame if they differ
	if result_bgr.shape[:2] != temp_vision_frame.shape[:2]:
		result_bgr = cv2.resize(result_bgr, (temp_vision_frame.shape[1], temp_vision_frame.shape[0]),
								interpolation=cv2.INTER_LINEAR)

	# Preserve alpha channel if present
	if temp_vision_frame.shape[2] == 4 and result_bgr.shape[2] == 3:
		result_bgra = numpy.dstack([result_bgr, temp_vision_frame[:, :, 3]])
		return result_bgra, temp_vision_mask

	return result_bgr, temp_vision_mask
