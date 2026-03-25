from configparser import ConfigParser
from typing import Any, Dict, List, Optional

from facefusion import state_manager
from facefusion.common_helper import cast_bool, cast_float, cast_int

CONFIG_PARSER = None
SAVEABLE_CONFIG_OPTIONS : Dict[str, List[str]] =\
{
	'face_detector':
	[
		'face_detector_model',
		'face_detector_size',
		'face_detector_margin',
		'face_detector_angles',
		'face_detector_score'
	],
	'face_landmarker':
	[
		'face_landmarker_model',
		'face_landmarker_score'
	],
	'face_selector':
	[
		'face_selector_mode',
		'face_selector_order',
		'face_selector_age_start',
		'face_selector_age_end',
		'face_selector_gender',
		'face_selector_race',
		'reference_face_position',
		'reference_face_distance',
		'reference_frame_number'
	],
	'face_masker':
	[
		'face_occluder_model',
		'face_parser_model',
		'face_mask_types',
		'face_mask_areas',
		'face_mask_regions',
		'face_mask_blur',
		'face_mask_padding'
	],
	'voice_extractor':
	[
		'voice_extractor_model'
	],
	'frame_extraction':
	[
		'trim_frame_start',
		'trim_frame_end',
		'temp_frame_format',
		'keep_temp'
	],
	'output_creation':
	[
		'output_image_quality',
		'output_image_scale',
		'output_audio_encoder',
		'output_audio_quality',
		'output_audio_volume',
		'output_video_encoder',
		'output_video_preset',
		'output_video_quality',
		'output_video_scale',
		'output_video_fps'
	],
	'processors':
	[
		'processors',
		'face_enhancer_model',
		'face_enhancer_blend',
		'face_enhancer_weight',
		'lip_syncer_model',
		'lip_syncer_pure_motion',
		'lip_syncer_motion_smoothing',
		'lip_syncer_motion_mask_mode',
		'lip_syncer_mask_blur',
		'lip_syncer_mask_erode',
		'lip_syncer_mask_expand',
		'lip_syncer_chin_expand',
		'lip_syncer_occlusion_dilate',
		'lip_syncer_occlusion_blur',
		'lip_syncer_expressiveness',
		'lip_syncer_weight'
	],
	'uis':
	[
		'open_browser',
		'ui_layouts',
		'ui_workflow'
	],
	'download':
	[
		'download_providers',
		'download_scope'
	],
	'benchmark':
	[
		'benchmark_mode',
		'benchmark_resolutions',
		'benchmark_cycle_count'
	],
	'execution':
	[
		'execution_device_ids',
		'execution_providers',
		'execution_thread_count'
	],
	'memory':
	[
		'video_memory_strategy',
		'system_memory_limit'
	],
	'misc':
	[
		'log_level',
		'halt_on_error'
	]
}


def get_config_parser() -> ConfigParser:
	global CONFIG_PARSER

	if CONFIG_PARSER is None:
		CONFIG_PARSER = ConfigParser()
		CONFIG_PARSER.read(state_manager.get_item('config_path'), encoding = 'utf-8')
	return CONFIG_PARSER


def clear_config_parser() -> None:
	global CONFIG_PARSER

	CONFIG_PARSER = None


def get_str_value(section : str, option : str, fallback : Optional[str] = None) -> Optional[str]:
	config_parser = get_config_parser()

	if config_parser.has_option(section, option) and config_parser.get(section, option).strip():
		return config_parser.get(section, option)
	return fallback


def get_int_value(section : str, option : str, fallback : Optional[str] = None) -> Optional[int]:
	config_parser = get_config_parser()

	if config_parser.has_option(section, option) and config_parser.get(section, option).strip():
		return config_parser.getint(section, option)
	return cast_int(fallback)


def get_float_value(section : str, option : str, fallback : Optional[str] = None) -> Optional[float]:
	config_parser = get_config_parser()

	if config_parser.has_option(section, option) and config_parser.get(section, option).strip():
		return config_parser.getfloat(section, option)
	return cast_float(fallback)


def get_bool_value(section : str, option : str, fallback : Optional[str] = None) -> Optional[bool]:
	config_parser = get_config_parser()

	if config_parser.has_option(section, option) and config_parser.get(section, option).strip():
		return config_parser.getboolean(section, option)
	return cast_bool(fallback)


def get_str_list(section : str, option : str, fallback : Optional[str] = None) -> Optional[List[str]]:
	config_parser = get_config_parser()

	if config_parser.has_option(section, option) and config_parser.get(section, option).strip():
		return config_parser.get(section, option).split()
	if fallback:
		return fallback.split()
	return None


def get_int_list(section : str, option : str, fallback : Optional[str] = None) -> Optional[List[int]]:
	config_parser = get_config_parser()

	if config_parser.has_option(section, option) and config_parser.get(section, option).strip():
		return list(map(int, config_parser.get(section, option).split()))
	if fallback:
		return list(map(int, fallback.split()))
	return None


def save_defaults() -> str:
	config_parser = get_config_parser()
	config_path = state_manager.get_item('config_path')

	for section, options in SAVEABLE_CONFIG_OPTIONS.items():
		if not config_parser.has_section(section):
			config_parser.add_section(section)

		for option in options:
			config_parser.set(section, option, serialize_value(state_manager.get_item(option)))

	with open(config_path, 'w', encoding = 'utf-8') as config_file:
		config_parser.write(config_file)

	clear_config_parser()
	return config_path


def serialize_value(value : Any) -> str:
	if value is None:
		return ''
	if isinstance(value, bool):
		return 'True' if value else 'False'
	if isinstance(value, (list, tuple)):
		return ' '.join(map(str, value))
	return str(value)
