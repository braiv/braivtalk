from argparse import ArgumentParser
from functools import lru_cache
from threading import Lock
from typing import Dict, List, Tuple

import cv2
import numpy

import facefusion.jobs.job_manager
import facefusion.jobs.job_store
from facefusion import config, content_analyser, face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, inference_manager, logger, state_manager, translator, video_manager, voice_extractor
from facefusion.audio import read_static_voice
from facefusion.common_helper import create_float_metavar, get_first
from facefusion.download import conditional_download_hashes, conditional_download_sources, resolve_download_url
from facefusion.face_analyser import get_many_faces, get_one_face, scale_face
from facefusion.face_helper import create_bounding_box, paste_back, warp_face_by_bounding_box, warp_face_by_face_landmark_5
from facefusion.face_masker import create_area_mask, create_box_mask, create_occlusion_mask
from facefusion.face_selector import select_faces
from facefusion.filesystem import filter_audio_paths, has_audio, resolve_relative_path
from facefusion.processors.live_portrait import create_rotation, limit_expression
from facefusion.processors.modules.lip_syncer import choices as lip_syncer_choices
from facefusion.processors.modules.lip_syncer.types import LipSyncerInputs, LipSyncerWeight
from facefusion.processors.types import LivePortraitExpression, LivePortraitFeatureVolume, LivePortraitMotionPoints, LivePortraitPitch, LivePortraitRoll, LivePortraitScale, LivePortraitTranslation, LivePortraitYaw, ProcessorOutputs
from facefusion.program_helper import find_argument_group
from facefusion.thread_helper import conditional_thread_semaphore, thread_semaphore
from facefusion.types import ApplyStateItem, Args, AudioFrame, BoundingBox, DownloadScope, DownloadSet, Face, InferencePool, ModelOptions, ModelSet, ProcessMode, VisionFrame
from facefusion.vision import read_static_image, read_static_video_frame

DRIVER_SMOOTH_SECONDS = 0.08
DRIVER_EMA_ALPHA = 0.65
DRIVER_MAX_STEP = 0.04
DRIVER_LIP_INDICES = [ 14, 17, 19, 20 ]
DRIVER_VARIANCE_RESTORE_MAX = 2.0
DRIVER_LIP_TIMING_BLEND = 0.35
_driver_expression_lock = Lock()
_driver_expression_cache : Dict[str, List[LivePortraitExpression]] = {}


@lru_cache()
def create_static_model_set(download_scope : DownloadScope) -> ModelSet:
	return\
	{
		'edtalk_256':
		{
			'__metadata__':
			{
				'vendor': 'tanshuai0219',
				'license': 'Apache-2.0',
				'year': 2024
			},
			'hashes':
			{
				'lip_syncer':
				{
					'url': resolve_download_url('models-3.3.0', 'edtalk_256.hash'),
					'path': resolve_relative_path('../.assets/models/edtalk_256.hash')
				}
			},
			'sources':
			{
				'lip_syncer':
				{
					'url': resolve_download_url('models-3.3.0', 'edtalk_256.onnx'),
					'path': resolve_relative_path('../.assets/models/edtalk_256.onnx')
				}
			},
			'type': 'edtalk',
			'size': (256, 256)
		},
		'wav2lip_96':
		{
			'__metadata__':
			{
				'vendor': 'Rudrabha',
				'license': 'Non-Commercial',
				'year': 2020
			},
			'hashes':
			{
				'lip_syncer':
				{
					'url': resolve_download_url('models-3.0.0', 'wav2lip_96.hash'),
					'path': resolve_relative_path('../.assets/models/wav2lip_96.hash')
				}
			},
			'sources':
			{
				'lip_syncer':
				{
					'url': resolve_download_url('models-3.0.0', 'wav2lip_96.onnx'),
					'path': resolve_relative_path('../.assets/models/wav2lip_96.onnx')
				}
			},
			'type': 'wav2lip',
			'size': (96, 96)
		},
		'wav2lip_gan_96':
		{
			'__metadata__':
			{
				'vendor': 'Rudrabha',
				'license': 'Non-Commercial',
				'year': 2020
			},
			'hashes':
			{
				'lip_syncer':
				{
					'url': resolve_download_url('models-3.0.0', 'wav2lip_gan_96.hash'),
					'path': resolve_relative_path('../.assets/models/wav2lip_gan_96.hash')
				}
			},
			'sources':
			{
				'lip_syncer':
				{
					'url': resolve_download_url('models-3.0.0', 'wav2lip_gan_96.onnx'),
					'path': resolve_relative_path('../.assets/models/wav2lip_gan_96.onnx')
				}
			},
			'type': 'wav2lip',
			'size': (96, 96)
		},
		'live_portrait':
		{
			'hashes':
			{
				'feature_extractor':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_feature_extractor.hash'),
					'path': resolve_relative_path('../.assets/models/live_portrait_feature_extractor.hash')
				},
				'motion_extractor':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_motion_extractor.hash'),
					'path': resolve_relative_path('../.assets/models/live_portrait_motion_extractor.hash')
				},
				'stitcher':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_stitcher.hash'),
					'path': resolve_relative_path('../.assets/models/live_portrait_stitcher.hash')
				},
				'generator':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_generator.hash'),
					'path': resolve_relative_path('../.assets/models/live_portrait_generator.hash')
				}
			},
			'sources':
			{
				'feature_extractor':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_feature_extractor.onnx'),
					'path': resolve_relative_path('../.assets/models/live_portrait_feature_extractor.onnx')
				},
				'motion_extractor':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_motion_extractor.onnx'),
					'path': resolve_relative_path('../.assets/models/live_portrait_motion_extractor.onnx')
				},
				'stitcher':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_stitcher.onnx'),
					'path': resolve_relative_path('../.assets/models/live_portrait_stitcher.onnx')
				},
				'generator':
				{
					'url': resolve_download_url('models-3.0.0', 'live_portrait_generator.onnx'),
					'path': resolve_relative_path('../.assets/models/live_portrait_generator.onnx')
				}
			}
		},
		'face_template':
		{
			'hashes':
			{
				'face_template':
				{
					'url': 'https://huggingface.co/bluefoxcreation/Templates/resolve/main/face-template/face_template.hash',
					'path': resolve_relative_path('../.assets/templates/face_template.hash')
				}
			},
			'sources':
			{
				'face_template':
				{
					'url': 'https://huggingface.co/bluefoxcreation/Templates/resolve/main/face-template/face_template.npy',
					'path': resolve_relative_path('../.assets/templates/face_template.npy')
				}
			}
		}
	}


def collect_model_downloads() -> Tuple[DownloadSet, DownloadSet]:
	model_set = create_static_model_set('full')
	model_hash_set = {}
	model_source_set = {}

	current_model = state_manager.get_item('lip_syncer_model')
	for lip_syncer_model in [ 'wav2lip_96', 'wav2lip_gan_96', 'edtalk_256' ]:
		if current_model == lip_syncer_model:
			model_hash_set['lip_syncer'] = model_set.get(lip_syncer_model).get('hashes').get('lip_syncer')
			model_source_set['lip_syncer'] = model_set.get(lip_syncer_model).get('sources').get('lip_syncer')

	if has_pure_motion():
		model_hashes = model_set.get('live_portrait').get('hashes')
		model_sources = model_set.get('live_portrait').get('sources')

		for model_hash in model_hashes.keys():
			model_hash_set[model_hash] = model_hashes.get(model_hash)

		for model_source in model_sources.keys():
			model_source_set[model_source] = model_sources.get(model_source)

	return model_hash_set, model_source_set


def get_inference_pool_model_names() -> List[str]:
	model_names = [ state_manager.get_item('lip_syncer_model') ]
	if has_pure_motion():
		model_names.append('live_portrait')
	return model_names


def get_inference_pool() -> InferencePool:
	model_names = get_inference_pool_model_names()
	_, model_source_set = collect_model_downloads()

	return inference_manager.get_inference_pool(__name__, model_names, model_source_set)


def clear_inference_pool() -> None:
	lip_syncer_model = state_manager.get_item('lip_syncer_model')

	# Clear both variants so toggling pure_motion cannot reuse a stale pool
	# that was created without the LivePortrait sessions.
	inference_manager.clear_inference_pool(__name__, [ lip_syncer_model ])
	inference_manager.clear_inference_pool(__name__, [ lip_syncer_model, 'live_portrait' ])
	clear_driver_expression_cache()


@lru_cache(maxsize = 1)
def get_static_face_template() -> Tuple[VisionFrame, BoundingBox]:
	face_template_path = get_face_template_options().get('sources').get('face_template').get('path')
	face_template = numpy.load(face_template_path)
	face_template_bounding_box = get_one_face(get_many_faces([ face_template ])).bounding_box
	return face_template, face_template_bounding_box


def get_model_options() -> ModelOptions:
	model_name = state_manager.get_item('lip_syncer_model')
	return create_static_model_set('full').get(model_name)


def get_face_template_options() -> ModelOptions:
	return create_static_model_set('full').get('face_template')


def register_args(program : ArgumentParser) -> None:
	group_processors = find_argument_group(program, 'processors')
	if group_processors:
		group_processors.add_argument('--lip-syncer-model', help = translator.get('help.model', __package__), default = config.get_str_value('processors', 'lip_syncer_model', 'wav2lip_gan_96'), choices = lip_syncer_choices.lip_syncer_models)
		group_processors.add_argument('--lip-syncer-pure-motion', help = translator.get('help.pure_motion', __package__), type = float, default = config.get_float_value('processors', 'lip_syncer_pure_motion', '0'), choices = lip_syncer_choices.lip_syncer_pure_motion_range, metavar = create_float_metavar(lip_syncer_choices.lip_syncer_pure_motion_range))
		group_processors.add_argument('--lip-syncer-motion-smoothing', help = translator.get('help.motion_smoothing', __package__), action = 'store_true', default = config.get_bool_value('processors', 'lip_syncer_motion_smoothing', 'False'))
		group_processors.add_argument('--lip-syncer-weight', help = translator.get('help.weight', __package__), type = float, default = config.get_float_value('processors', 'lip_syncer_weight', '0.5'), choices = lip_syncer_choices.lip_syncer_weight_range, metavar = create_float_metavar(lip_syncer_choices.lip_syncer_weight_range))
		facefusion.jobs.job_store.register_step_keys([ 'lip_syncer_model', 'lip_syncer_pure_motion', 'lip_syncer_motion_smoothing', 'lip_syncer_weight' ])


def apply_args(args : Args, apply_state_item : ApplyStateItem) -> None:
	apply_state_item('lip_syncer_model', args.get('lip_syncer_model'))
	apply_state_item('lip_syncer_pure_motion', args.get('lip_syncer_pure_motion'))
	apply_state_item('lip_syncer_motion_smoothing', args.get('lip_syncer_motion_smoothing'))
	apply_state_item('lip_syncer_weight', args.get('lip_syncer_weight'))


def pre_check() -> bool:
	model_hash_set, model_source_set = collect_model_downloads()
	face_template_hash_set = get_face_template_options().get('hashes')
	face_template_source_set = get_face_template_options().get('sources')

	return conditional_download_hashes(model_hash_set) and conditional_download_sources(model_source_set) and conditional_download_hashes(face_template_hash_set) and conditional_download_sources(face_template_source_set)


def pre_process(mode : ProcessMode) -> bool:
	if not has_audio(state_manager.get_item('source_paths')):
		logger.error(translator.get('choose_audio_source') + translator.get('exclamation_mark'), __name__)
		return False
	return True


def post_process() -> None:
	read_static_image.cache_clear()
	read_static_video_frame.cache_clear()
	read_static_voice.cache_clear()
	clear_driver_expression_cache()
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
		voice_extractor.clear_inference_pool()


def has_pure_motion() -> bool:
	pure_motion = state_manager.get_item('lip_syncer_pure_motion')
	return pure_motion is not None and pure_motion > 0


def has_motion_smoothing() -> bool:
	return has_pure_motion() and bool(state_manager.get_item('lip_syncer_motion_smoothing'))


def clear_driver_expression_cache() -> None:
	with _driver_expression_lock:
		_driver_expression_cache.clear()


def sync_lip(target_face : Face, source_voice_frame : AudioFrame, temp_vision_frame : VisionFrame, frame_number : int = 0) -> VisionFrame:
	source_voice_frame = prepare_audio_frame(source_voice_frame)
	crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, target_face.landmark_set.get('5/68'), 'ffhq_512', (512, 512))
	crop_masks = create_occlusion_masks(crop_vision_frame)

	if has_pure_motion():
		crop_vision_frame, additional_masks = process_live_portrait_motion(target_face, source_voice_frame, crop_vision_frame, affine_matrix, frame_number)
		crop_masks.extend(additional_masks)
	else:
		crop_vision_frame, additional_masks = process_standard_lip_sync(target_face, source_voice_frame, crop_vision_frame, affine_matrix)
		crop_masks.extend(additional_masks)

	crop_mask = numpy.minimum.reduce(crop_masks)
	paste_vision_frame = paste_back(temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix)
	return paste_vision_frame


def create_occlusion_masks(crop_vision_frame : VisionFrame) -> List:
	crop_masks = []
	if 'occlusion' in state_manager.get_item('face_mask_types'):
		occlusion_mask = create_occlusion_mask(crop_vision_frame)
		crop_masks.append(occlusion_mask)
	return crop_masks


def process_standard_lip_sync(target_face : Face, source_voice_frame : AudioFrame, crop_vision_frame : VisionFrame, affine_matrix) -> Tuple[VisionFrame, List]:
	model_name = state_manager.get_item('lip_syncer_model')
	face_landmark_68 = cv2.transform(target_face.landmark_set.get('68').reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
	bounding_box = create_bounding_box(face_landmark_68)
	crop_vision_frame = apply_lip_syncer(crop_vision_frame, source_voice_frame, bounding_box)

	masks = []
	if model_name == 'edtalk_256':
		box_mask = create_box_mask(crop_vision_frame, state_manager.get_item('face_mask_blur'), state_manager.get_item('face_mask_padding'))
		masks.append(box_mask)
	if model_name.startswith('wav2lip'):
		area_mask = create_area_mask(crop_vision_frame, face_landmark_68, [ 'lower-face' ])
		masks.append(area_mask)
	return crop_vision_frame, masks


def apply_lip_syncer(crop_vision_frame : VisionFrame, source_voice_frame : AudioFrame, bounding_box : BoundingBox) -> VisionFrame:
	model_name = state_manager.get_item('lip_syncer_model')
	model_size = get_model_options().get('size')

	if model_name == 'edtalk_256':
		lip_syncer_weight = numpy.array([ state_manager.get_item('lip_syncer_weight') ]).astype(numpy.float32) * 1.25
		crop_vision_frame = prepare_crop_frame(crop_vision_frame)
		crop_vision_frame = forward_edtalk(source_voice_frame, crop_vision_frame, lip_syncer_weight)
		crop_vision_frame = normalize_crop_frame(crop_vision_frame)
	if model_name.startswith('wav2lip'):
		bounding_box = resize_bounding_box(bounding_box, 1 / 8)
		area_vision_frame, area_matrix = warp_face_by_bounding_box(crop_vision_frame, bounding_box, model_size)
		area_vision_frame = prepare_crop_frame(area_vision_frame)
		area_vision_frame = forward_wav2lip(source_voice_frame, area_vision_frame)
		area_vision_frame = normalize_crop_frame(area_vision_frame)
		crop_vision_frame = cv2.warpAffine(area_vision_frame, cv2.invertAffineTransform(area_matrix), (512, 512), borderMode = cv2.BORDER_REPLICATE)
	return crop_vision_frame


def process_live_portrait_motion(target_face : Face, source_voice_frame : AudioFrame, crop_vision_frame : VisionFrame, affine_matrix, frame_number : int = 0) -> Tuple[VisionFrame, List]:
	face_template_expression = extract_template_expression(source_voice_frame, frame_number)
	crop_vision_frame = prepare_refine_frame(crop_vision_frame)
	feature_volume = forward_extract_feature(crop_vision_frame)
	pitch, yaw, roll, scale, translation, expression, motion_points = forward_extract_motion(crop_vision_frame)

	motion_points_target = calculate_target_motion_points(pitch, yaw, roll, scale, translation, expression, motion_points)
	blended_expression = create_blended_expression(expression, face_template_expression)
	motion_points_source = calculate_source_motion_points(pitch, yaw, roll, scale, translation, blended_expression, motion_points)
	motion_points_source = forward_stitch_motion_points(motion_points_source, motion_points_target)

	crop_vision_frame = forward_generate_frame(feature_volume, motion_points_source, motion_points_target)
	crop_vision_frame = normalize_refine_frame(crop_vision_frame)
	box_mask = create_box_mask(crop_vision_frame, state_manager.get_item('face_mask_blur'), state_manager.get_item('face_mask_padding'))
	return crop_vision_frame, [ box_mask ]


def extract_template_expression(source_voice_frame : AudioFrame, frame_number : int = 0) -> LivePortraitExpression:
	if has_motion_smoothing():
		driver_expression_sequence = get_driver_expression_sequence()
		if driver_expression_sequence:
			frame_index = min(max(frame_number, 0), len(driver_expression_sequence) - 1)
			return driver_expression_sequence[frame_index].copy()
	return extract_template_expression_for_frame(source_voice_frame)


def extract_template_expression_for_frame(source_voice_frame : AudioFrame) -> LivePortraitExpression:
	face_template_frame, face_template_bounding_box = get_static_face_template()
	face_template_crop_frame = apply_lip_syncer(face_template_frame.copy(), source_voice_frame, face_template_bounding_box.copy())
	face_template_crop_frame = prepare_refine_frame(face_template_crop_frame)
	face_template_expression = forward_extract_motion(face_template_crop_frame)[5]
	face_template_expression *= state_manager.get_item('lip_syncer_pure_motion')
	return face_template_expression


def get_driver_expression_sequence() -> List[LivePortraitExpression]:
	source_audio_path = get_first(filter_audio_paths(state_manager.get_item('source_paths')))
	output_video_fps = state_manager.get_item('output_video_fps')
	if not source_audio_path or not output_video_fps:
		return []

	cache_key = create_driver_expression_sequence_cache_key(source_audio_path, output_video_fps)
	with _driver_expression_lock:
		if cache_key not in _driver_expression_cache:
			_driver_expression_cache[cache_key] = build_driver_expression_sequence(source_audio_path, output_video_fps)
		return _driver_expression_cache[cache_key]


def create_driver_expression_sequence_cache_key(source_audio_path : str, output_video_fps : float) -> str:
	return ':'.join(
	[
		source_audio_path,
		str(output_video_fps),
		str(state_manager.get_item('lip_syncer_model')),
		str(state_manager.get_item('lip_syncer_pure_motion')),
		str(state_manager.get_item('lip_syncer_weight'))
	])


def build_driver_expression_sequence(source_audio_path : str, output_video_fps : float) -> List[LivePortraitExpression]:
	voice_frames = read_static_voice(source_audio_path, output_video_fps)
	if not voice_frames:
		return []

	driver_expression_sequence = []
	for source_voice_frame in voice_frames:
		driver_expression_sequence.append(extract_template_expression_for_frame(prepare_audio_frame(source_voice_frame)))

	return restore_driver_amplitude(driver_expression_sequence, stabilize_sequence(driver_expression_sequence))


def stabilize_sequence(driver_expression_sequence : List[LivePortraitExpression]) -> List[LivePortraitExpression]:
	if len(driver_expression_sequence) < 3:
		return [ driver_expression.copy() for driver_expression in driver_expression_sequence ]

	output_video_fps = state_manager.get_item('output_video_fps') or 25.0
	window_radius = max(1, int(round(output_video_fps * DRIVER_SMOOTH_SECONDS)))
	driver_expression_array = numpy.stack(driver_expression_sequence).astype(numpy.float32)
	smoothed_array = numpy.empty_like(driver_expression_array)

	for index in range(len(driver_expression_array)):
		window_start = max(0, index - window_radius)
		window_end = min(len(driver_expression_array), index + window_radius + 1)
		window = driver_expression_array[window_start:window_end]
		window_mean = numpy.mean(window, axis = 0)
		window_median = numpy.median(window, axis = 0)
		smoothed_array[index] = (window_mean + window_median) / 2

	ema_array = smoothed_array.copy()
	for index in range(1, len(ema_array)):
		ema_array[index] = DRIVER_EMA_ALPHA * smoothed_array[index] + (1 - DRIVER_EMA_ALPHA) * ema_array[index - 1]

	clamped_array = ema_array.copy()
	for index in range(1, len(clamped_array)):
		step = clamped_array[index] - clamped_array[index - 1]
		step = numpy.clip(step, -DRIVER_MAX_STEP, DRIVER_MAX_STEP)
		clamped_array[index] = clamped_array[index - 1] + step

	return [ driver_expression.copy() for driver_expression in clamped_array ]


def restore_driver_amplitude(original_sequence : List[LivePortraitExpression], smoothed_sequence : List[LivePortraitExpression]) -> List[LivePortraitExpression]:
	if len(original_sequence) != len(smoothed_sequence) or not original_sequence:
		return smoothed_sequence

	original_array = numpy.stack(original_sequence).astype(numpy.float32)
	restored_array = numpy.stack(smoothed_sequence).astype(numpy.float32)

	for lip_index in DRIVER_LIP_INDICES:
		original_lip_motion = original_array[:, 0, lip_index, :]
		smoothed_lip_motion = restored_array[:, 0, lip_index, :]
		original_mean = numpy.mean(original_lip_motion, axis = 0, keepdims = True)
		smoothed_mean = numpy.mean(smoothed_lip_motion, axis = 0, keepdims = True)
		original_std = numpy.std(original_lip_motion, axis = 0)
		smoothed_std = numpy.maximum(numpy.std(smoothed_lip_motion, axis = 0), 1e-6)
		restore_scale = numpy.clip(original_std / smoothed_std, 1.0, DRIVER_VARIANCE_RESTORE_MAX)
		restored_lip_motion = smoothed_mean + (smoothed_lip_motion - smoothed_mean) * restore_scale
		restored_lip_motion = restored_lip_motion * (1 - DRIVER_LIP_TIMING_BLEND) + (original_lip_motion - original_mean + smoothed_mean) * DRIVER_LIP_TIMING_BLEND
		restored_array[:, 0, lip_index, :] = restored_lip_motion

	return [ driver_expression.copy() for driver_expression in restored_array ]


def calculate_target_motion_points(pitch, yaw, roll, scale, translation, expression, motion_points) -> LivePortraitMotionPoints:
	rotation = create_rotation(pitch, yaw, roll)
	return scale * (motion_points @ rotation.T + expression) + translation


def calculate_source_motion_points(pitch, yaw, roll, scale, translation, expression, motion_points) -> LivePortraitMotionPoints:
	rotation = create_rotation(pitch, yaw, roll)
	motion_points_source = motion_points @ rotation.T
	motion_points_source += expression
	motion_points_source *= scale
	motion_points_source += translation
	return motion_points_source


def create_blended_expression(expression : LivePortraitExpression, face_template_expression : LivePortraitExpression) -> LivePortraitExpression:
	temp_expression = expression.copy()
	temp_expression = blend_expression(6, temp_expression, face_template_expression, 0.5, 0.5, 0.5)
	temp_expression = blend_expression(12, temp_expression, face_template_expression, 0.5, 0.5, 0.5)
	temp_expression = blend_expression(14, temp_expression, face_template_expression, 0.6, 0.7, 0.7)
	temp_expression = blend_expression(17, temp_expression, face_template_expression, 0.5, 0.8, 0.7)
	temp_expression = blend_expression(20, temp_expression, face_template_expression, 0.5, 0.6, 0.7)
	lip_open = float(numpy.interp(face_template_expression[0, 19, 1].clip(0, 1), [0, 1], [0.9, 1.5]))
	temp_expression = blend_expression(19, temp_expression, face_template_expression, 0.5, lip_open, 0.85)
	temp_expression = limit_expression(temp_expression)
	return temp_expression


def blend_expression(index : int, temp_expression : LivePortraitExpression, face_template_expression : LivePortraitExpression, blend_x : float, blend_y : float, blend_z : float) -> LivePortraitExpression:
	temp_expression[0, index, 0] = temp_expression[0, index, 0] * max(1 - blend_x, 0) + face_template_expression[0, index, 0] * blend_x
	temp_expression[0, index, 1] = temp_expression[0, index, 1] * max(1 - blend_y, 0) + face_template_expression[0, index, 1] * blend_y
	temp_expression[0, index, 2] = temp_expression[0, index, 2] * max(1 - blend_z, 0) + face_template_expression[0, index, 2] * blend_z
	return temp_expression


def forward_edtalk(temp_audio_frame : AudioFrame, crop_vision_frame : VisionFrame, lip_syncer_weight : LipSyncerWeight) -> VisionFrame:
	lip_syncer = get_inference_pool().get('lip_syncer')

	with conditional_thread_semaphore():
		crop_vision_frame = lip_syncer.run(None,
		{
			'source': temp_audio_frame,
			'target': crop_vision_frame,
			'weight': lip_syncer_weight
		})[0]

	return crop_vision_frame


def forward_wav2lip(temp_audio_frame : AudioFrame, area_vision_frame : VisionFrame) -> VisionFrame:
	lip_syncer = get_inference_pool().get('lip_syncer')

	with conditional_thread_semaphore():
		area_vision_frame = lip_syncer.run(None,
		{
			'source': temp_audio_frame,
			'target': area_vision_frame
		})[0]

	return area_vision_frame


def forward_extract_feature(crop_vision_frame : VisionFrame) -> LivePortraitFeatureVolume:
	feature_extractor = get_inference_pool().get('feature_extractor')

	with conditional_thread_semaphore():
		feature_volume = feature_extractor.run(None,
		{
			'input': crop_vision_frame
		})[0]

	return feature_volume


def forward_extract_motion(crop_vision_frame : VisionFrame) -> Tuple[LivePortraitPitch, LivePortraitYaw, LivePortraitRoll, LivePortraitScale, LivePortraitTranslation, LivePortraitExpression, LivePortraitMotionPoints]:
	motion_extractor = get_inference_pool().get('motion_extractor')

	with conditional_thread_semaphore():
		pitch, yaw, roll, scale, translation, expression, motion_points = motion_extractor.run(None,
		{
			'input': crop_vision_frame
		})

	return pitch, yaw, roll, scale, translation, expression, motion_points


def forward_stitch_motion_points(source_motion_points : LivePortraitMotionPoints, target_motion_points : LivePortraitMotionPoints) -> LivePortraitMotionPoints:
	stitcher = get_inference_pool().get('stitcher')

	with thread_semaphore():
		motion_points = stitcher.run(None,
		{
			'source': source_motion_points,
			'target': target_motion_points
		})[0]

	return motion_points


def forward_generate_frame(feature_volume : LivePortraitFeatureVolume, source_motion_points : LivePortraitMotionPoints, target_motion_points : LivePortraitMotionPoints) -> VisionFrame:
	generator = get_inference_pool().get('generator')

	with thread_semaphore():
		crop_vision_frame = generator.run(None,
		{
			'feature_volume': feature_volume,
			'source': source_motion_points,
			'target': target_motion_points
		})[0][0]

	return crop_vision_frame


def prepare_audio_frame(temp_audio_frame : AudioFrame) -> AudioFrame:
	model_type = get_model_options().get('type')
	temp_audio_frame = numpy.maximum(numpy.exp(-5 * numpy.log(10)), temp_audio_frame)
	temp_audio_frame = numpy.log10(temp_audio_frame) * 1.6 + 3.2
	temp_audio_frame = temp_audio_frame.clip(-4, 4).astype(numpy.float32)

	if model_type == 'wav2lip':
		temp_audio_frame = temp_audio_frame * state_manager.get_item('lip_syncer_weight') * 2.0

	temp_audio_frame = numpy.expand_dims(temp_audio_frame, axis = (0, 1))
	return temp_audio_frame


def prepare_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	model_type = get_model_options().get('type')
	model_size = get_model_options().get('size')

	if model_type == 'edtalk':
		crop_vision_frame = cv2.resize(crop_vision_frame, model_size, interpolation = cv2.INTER_AREA)
		crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
		crop_vision_frame = numpy.expand_dims(crop_vision_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)

	if model_type == 'wav2lip':
		crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis = 0)
		prepare_vision_frame = crop_vision_frame.copy()
		prepare_vision_frame[:, model_size[0] // 2:] = 0
		crop_vision_frame = numpy.concatenate((prepare_vision_frame, crop_vision_frame), axis = 3)
		crop_vision_frame = crop_vision_frame.transpose(0, 3, 1, 2).astype(numpy.float32) / 255.0

	return crop_vision_frame


def prepare_refine_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	crop_vision_frame = cv2.resize(crop_vision_frame, (256, 256), interpolation = cv2.INTER_CUBIC)
	crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
	crop_vision_frame = numpy.expand_dims(crop_vision_frame.transpose(2, 0, 1), axis = 0).astype(numpy.float32)
	return crop_vision_frame


def resize_bounding_box(bounding_box : BoundingBox, aspect_ratio : float) -> BoundingBox:
	x1, y1, x2, y2 = bounding_box
	y1 -= numpy.abs(y2 - y1) * aspect_ratio
	return numpy.array([ x1, y1, x2, y2 ])


def normalize_crop_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	model_type = get_model_options().get('type')
	crop_vision_frame = crop_vision_frame[0].transpose(1, 2, 0)
	crop_vision_frame = crop_vision_frame.clip(0, 1) * 255
	crop_vision_frame = crop_vision_frame.astype(numpy.uint8)

	if model_type == 'edtalk':
		crop_vision_frame = crop_vision_frame[:, :, ::-1]
		crop_vision_frame = cv2.resize(crop_vision_frame, (512, 512), interpolation = cv2.INTER_CUBIC)

	return crop_vision_frame


def normalize_refine_frame(crop_vision_frame : VisionFrame) -> VisionFrame:
	crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
	crop_vision_frame = crop_vision_frame.clip(0, 1) * 255
	crop_vision_frame = crop_vision_frame.astype(numpy.uint8)
	crop_vision_frame = crop_vision_frame[:, :, ::-1]
	crop_vision_frame = cv2.resize(crop_vision_frame, (512, 512), interpolation = cv2.INTER_CUBIC)
	return crop_vision_frame


def process_frame(inputs : LipSyncerInputs) -> ProcessorOutputs:
	reference_vision_frame = inputs.get('reference_vision_frame')
	source_voice_frame = inputs.get('source_voice_frame')
	target_vision_frame = inputs.get('target_vision_frame')
	temp_vision_frame = inputs.get('temp_vision_frame')
	temp_vision_mask = inputs.get('temp_vision_mask')
	frame_number = inputs.get('frame_number')
	target_faces = select_faces(reference_vision_frame, target_vision_frame)

	if target_faces:
		for target_face in target_faces:
			target_face = scale_face(target_face, target_vision_frame, temp_vision_frame)
			temp_vision_frame = sync_lip(target_face, source_voice_frame, temp_vision_frame, frame_number)

	return temp_vision_frame, temp_vision_mask
