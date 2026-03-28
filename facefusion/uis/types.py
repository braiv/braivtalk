from typing import Any, Dict, IO, Literal, TypeAlias

File : TypeAlias = IO[Any]
ComponentName = Literal\
[
	'face_detector_angles_checkbox_group',
	'face_detector_model_dropdown',
	'face_detector_margin_slider',
	'face_detector_score_slider',
	'face_detector_size_dropdown',
	'face_enhancer_blend_slider',
	'face_enhancer_model_dropdown',
	'face_enhancer_weight_slider',
	'face_landmarker_model_dropdown',
	'face_landmarker_score_slider',
	'face_mask_occlusion_checkbox',
	'face_mask_blur_slider',
	'face_mask_padding_bottom_slider',
	'face_mask_padding_left_slider',
	'face_mask_padding_right_slider',
	'face_mask_padding_top_slider',
	'face_selector_age_range_slider',
	'face_selector_gender_dropdown',
	'face_selector_mode_dropdown',
	'face_selector_order_dropdown',
	'face_selector_race_dropdown',
	'face_occluder_model_dropdown',
	'face_parser_model_dropdown',
	'voice_extractor_model_dropdown',
	'job_list_job_status_checkbox_group',
	'lip_syncer_pipeline_dropdown',
	'lip_syncer_model_dropdown',
	'lip_syncer_pure_motion_slider',
	'lip_syncer_motion_damping_slider',
	'lip_syncer_crop_stabilization_slider',
	'lip_syncer_motion_smoothing_checkbox',
	'lip_syncer_motion_mask_mode_dropdown',
	'lip_syncer_mask_blur_slider',
	'lip_syncer_mask_erode_slider',
	'lip_syncer_mask_expand_slider',
	'lip_syncer_chin_expand_slider',
	'lip_syncer_occlusion_dilate_slider',
	'lip_syncer_occlusion_blur_slider',
	'lip_syncer_expressiveness_slider',
	'lip_syncer_weight_slider',
	'ditto_source_mode_dropdown',
	'ditto_backend_dropdown',
	'ditto_render_mode_dropdown',
	'ditto_registration_crop_mode_dropdown',
	'ditto_live_crop_mode_dropdown',
	'ditto_composite_geometry_mode_dropdown',
	'output_image',
	'output_video',
	'output_video_fps_slider',
	'preview_image',
	'preview_frame_slider',
	'preview_mode_dropdown',
	'preview_resolution_dropdown',
	'processors_checkbox_group',
	'reference_face_distance_slider',
	'reference_face_position_gallery',
	'source_audio',
	'source_image',
	'target_image',
	'target_video',
	'ui_workflow_dropdown',
	'webcam_device_id_dropdown',
	'webcam_fps_slider',
	'webcam_mode_radio',
	'webcam_resolution_dropdown'
]
Component : TypeAlias = Any
ComponentOptions : TypeAlias = Dict[str, Any]

JobManagerAction = Literal['job-create', 'job-submit', 'job-delete', 'job-add-step', 'job-remix-step', 'job-insert-step', 'job-remove-step']
JobRunnerAction = Literal['job-run', 'job-run-all', 'job-retry', 'job-retry-all']

PreviewMode = Literal[ 'default', 'frame-by-frame', 'face-by-face', 'mask-debug' ]

MockArgs : TypeAlias = Any
