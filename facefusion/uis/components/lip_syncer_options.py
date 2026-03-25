from typing import List, Optional, Tuple

import facefusion.choices
import gradio

from facefusion import state_manager, translator
from facefusion.common_helper import calculate_float_step, calculate_int_step
from facefusion.processors.core import load_processor_module
from facefusion.processors.modules.lip_syncer import choices as lip_syncer_choices
from facefusion.processors.modules.lip_syncer.types import LipSyncerModel, LipSyncerMotionMaskMode, LipSyncerWeight
from facefusion.sanitizer import sanitize_int_range
from facefusion.uis.core import get_ui_component, register_ui_component

LIP_SYNCER_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
LIP_SYNCER_PURE_MOTION_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX : Optional[gradio.Checkbox] = None
LIP_SYNCER_MOTION_MASK_MODE_DROPDOWN : Optional[gradio.Dropdown] = None
LIP_SYNCER_MASK_BLUR_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_MASK_ERODE_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_MASK_EXPAND_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_CHIN_EXPAND_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_OCCLUSION_DILATE_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_OCCLUSION_BLUR_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_MASK_PADDING_TOP_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_MASK_PADDING_RIGHT_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_MASK_PADDING_BOTTOM_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_MASK_PADDING_LEFT_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_EXPRESSIVENESS_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_WEIGHT_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global LIP_SYNCER_MODEL_DROPDOWN
	global LIP_SYNCER_PURE_MOTION_SLIDER
	global LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX
	global LIP_SYNCER_MOTION_MASK_MODE_DROPDOWN
	global LIP_SYNCER_MASK_BLUR_SLIDER
	global LIP_SYNCER_MASK_ERODE_SLIDER
	global LIP_SYNCER_MASK_EXPAND_SLIDER
	global LIP_SYNCER_CHIN_EXPAND_SLIDER
	global LIP_SYNCER_OCCLUSION_DILATE_SLIDER
	global LIP_SYNCER_OCCLUSION_BLUR_SLIDER
	global LIP_SYNCER_MASK_PADDING_TOP_SLIDER
	global LIP_SYNCER_MASK_PADDING_RIGHT_SLIDER
	global LIP_SYNCER_MASK_PADDING_BOTTOM_SLIDER
	global LIP_SYNCER_MASK_PADDING_LEFT_SLIDER
	global LIP_SYNCER_EXPRESSIVENESS_SLIDER
	global LIP_SYNCER_WEIGHT_SLIDER

	has_lip_syncer = 'lip_syncer' in state_manager.get_item('processors')
	LIP_SYNCER_MODEL_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.model_dropdown', 'facefusion.processors.modules.lip_syncer'),
		choices = lip_syncer_choices.lip_syncer_models,
		value = state_manager.get_item('lip_syncer_model'),
		visible = has_lip_syncer
	)
	LIP_SYNCER_PURE_MOTION_SLIDER = gradio.Slider(
		label = translator.get('uis.pure_motion_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_pure_motion'),
		step = calculate_float_step(lip_syncer_choices.lip_syncer_pure_motion_range),
		minimum = lip_syncer_choices.lip_syncer_pure_motion_range[0],
		maximum = lip_syncer_choices.lip_syncer_pure_motion_range[-1],
		visible = has_lip_syncer
	)
	LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX = gradio.Checkbox(
		label = translator.get('uis.motion_smoothing_checkbox', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_motion_smoothing'),
		visible = has_lip_syncer
	)
	LIP_SYNCER_MOTION_MASK_MODE_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.motion_mask_mode_dropdown', 'facefusion.processors.modules.lip_syncer'),
		choices = lip_syncer_choices.lip_syncer_motion_mask_modes,
		value = state_manager.get_item('lip_syncer_motion_mask_mode'),
		visible = has_lip_syncer
	)
	LIP_SYNCER_MASK_BLUR_SLIDER = gradio.Slider(
		label = translator.get('uis.mask_blur_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_mask_blur'),
		step = calculate_float_step(lip_syncer_choices.lip_syncer_mask_blur_range),
		minimum = lip_syncer_choices.lip_syncer_mask_blur_range[0],
		maximum = lip_syncer_choices.lip_syncer_mask_blur_range[-1],
		visible = has_lip_syncer
	)
	LIP_SYNCER_MASK_ERODE_SLIDER = gradio.Slider(
		label = translator.get('uis.mask_erode_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_mask_erode'),
		step = calculate_int_step(lip_syncer_choices.lip_syncer_mask_erode_range),
		minimum = lip_syncer_choices.lip_syncer_mask_erode_range[0],
		maximum = lip_syncer_choices.lip_syncer_mask_erode_range[-1],
		visible = has_lip_syncer
	)
	LIP_SYNCER_MASK_EXPAND_SLIDER = gradio.Slider(
		label = translator.get('uis.mask_expand_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_mask_expand'),
		step = calculate_int_step(lip_syncer_choices.lip_syncer_mask_expand_range),
		minimum = lip_syncer_choices.lip_syncer_mask_expand_range[0],
		maximum = lip_syncer_choices.lip_syncer_mask_expand_range[-1],
		visible = has_lip_syncer
	)
	LIP_SYNCER_CHIN_EXPAND_SLIDER = gradio.Slider(
		label = translator.get('uis.chin_expand_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_chin_expand'),
		step = calculate_int_step(lip_syncer_choices.lip_syncer_chin_expand_range),
		minimum = lip_syncer_choices.lip_syncer_chin_expand_range[0],
		maximum = lip_syncer_choices.lip_syncer_chin_expand_range[-1],
		visible = has_lip_syncer
	)
	LIP_SYNCER_OCCLUSION_DILATE_SLIDER = gradio.Slider(
		label = translator.get('uis.occlusion_dilate_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_occlusion_dilate'),
		step = calculate_int_step(lip_syncer_choices.lip_syncer_occlusion_dilate_range),
		minimum = lip_syncer_choices.lip_syncer_occlusion_dilate_range[0],
		maximum = lip_syncer_choices.lip_syncer_occlusion_dilate_range[-1],
		visible = has_lip_syncer
	)
	LIP_SYNCER_OCCLUSION_BLUR_SLIDER = gradio.Slider(
		label = translator.get('uis.occlusion_blur_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_occlusion_blur'),
		step = calculate_float_step(lip_syncer_choices.lip_syncer_occlusion_blur_range),
		minimum = lip_syncer_choices.lip_syncer_occlusion_blur_range[0],
		maximum = lip_syncer_choices.lip_syncer_occlusion_blur_range[-1],
		visible = has_lip_syncer
	)
	with gradio.Row():
		LIP_SYNCER_MASK_PADDING_TOP_SLIDER = gradio.Slider(
			label = translator.get('uis.face_mask_padding_top_slider'),
			value = state_manager.get_item('face_mask_padding')[0],
			step = calculate_int_step(facefusion.choices.face_mask_padding_range),
			minimum = facefusion.choices.face_mask_padding_range[0],
			maximum = facefusion.choices.face_mask_padding_range[-1],
			visible = has_lip_syncer
		)
		LIP_SYNCER_MASK_PADDING_RIGHT_SLIDER = gradio.Slider(
			label = translator.get('uis.face_mask_padding_right_slider'),
			value = state_manager.get_item('face_mask_padding')[1],
			step = calculate_int_step(facefusion.choices.face_mask_padding_range),
			minimum = facefusion.choices.face_mask_padding_range[0],
			maximum = facefusion.choices.face_mask_padding_range[-1],
			visible = has_lip_syncer
		)
	with gradio.Row():
		LIP_SYNCER_MASK_PADDING_BOTTOM_SLIDER = gradio.Slider(
			label = translator.get('uis.face_mask_padding_bottom_slider'),
			value = state_manager.get_item('face_mask_padding')[2],
			step = calculate_int_step(facefusion.choices.face_mask_padding_range),
			minimum = facefusion.choices.face_mask_padding_range[0],
			maximum = facefusion.choices.face_mask_padding_range[-1],
			visible = has_lip_syncer
		)
		LIP_SYNCER_MASK_PADDING_LEFT_SLIDER = gradio.Slider(
			label = translator.get('uis.face_mask_padding_left_slider'),
			value = state_manager.get_item('face_mask_padding')[3],
			step = calculate_int_step(facefusion.choices.face_mask_padding_range),
			minimum = facefusion.choices.face_mask_padding_range[0],
			maximum = facefusion.choices.face_mask_padding_range[-1],
			visible = has_lip_syncer
		)
	LIP_SYNCER_EXPRESSIVENESS_SLIDER = gradio.Slider(
		label = translator.get('uis.expressiveness_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_expressiveness'),
		step = calculate_float_step(lip_syncer_choices.lip_syncer_expressiveness_range),
		minimum = lip_syncer_choices.lip_syncer_expressiveness_range[0],
		maximum = lip_syncer_choices.lip_syncer_expressiveness_range[-1],
		visible = has_lip_syncer
	)
	LIP_SYNCER_WEIGHT_SLIDER = gradio.Slider(
		label = translator.get('uis.weight_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_weight'),
		step = calculate_float_step(lip_syncer_choices.lip_syncer_weight_range),
		minimum = lip_syncer_choices.lip_syncer_weight_range[0],
		maximum = lip_syncer_choices.lip_syncer_weight_range[-1],
		visible = has_lip_syncer
	)
	register_ui_component('lip_syncer_model_dropdown', LIP_SYNCER_MODEL_DROPDOWN)
	register_ui_component('lip_syncer_pure_motion_slider', LIP_SYNCER_PURE_MOTION_SLIDER)
	register_ui_component('lip_syncer_motion_smoothing_checkbox', LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX)
	register_ui_component('lip_syncer_motion_mask_mode_dropdown', LIP_SYNCER_MOTION_MASK_MODE_DROPDOWN)
	register_ui_component('lip_syncer_mask_blur_slider', LIP_SYNCER_MASK_BLUR_SLIDER)
	register_ui_component('lip_syncer_mask_erode_slider', LIP_SYNCER_MASK_ERODE_SLIDER)
	register_ui_component('lip_syncer_mask_expand_slider', LIP_SYNCER_MASK_EXPAND_SLIDER)
	register_ui_component('lip_syncer_chin_expand_slider', LIP_SYNCER_CHIN_EXPAND_SLIDER)
	register_ui_component('lip_syncer_occlusion_dilate_slider', LIP_SYNCER_OCCLUSION_DILATE_SLIDER)
	register_ui_component('lip_syncer_occlusion_blur_slider', LIP_SYNCER_OCCLUSION_BLUR_SLIDER)
	register_ui_component('face_mask_padding_top_slider', LIP_SYNCER_MASK_PADDING_TOP_SLIDER)
	register_ui_component('face_mask_padding_right_slider', LIP_SYNCER_MASK_PADDING_RIGHT_SLIDER)
	register_ui_component('face_mask_padding_bottom_slider', LIP_SYNCER_MASK_PADDING_BOTTOM_SLIDER)
	register_ui_component('face_mask_padding_left_slider', LIP_SYNCER_MASK_PADDING_LEFT_SLIDER)
	register_ui_component('lip_syncer_expressiveness_slider', LIP_SYNCER_EXPRESSIVENESS_SLIDER)
	register_ui_component('lip_syncer_weight_slider', LIP_SYNCER_WEIGHT_SLIDER)


def listen() -> None:
	LIP_SYNCER_MODEL_DROPDOWN.change(update_lip_syncer_model, inputs = LIP_SYNCER_MODEL_DROPDOWN, outputs = LIP_SYNCER_MODEL_DROPDOWN)
	LIP_SYNCER_PURE_MOTION_SLIDER.release(update_lip_syncer_pure_motion, inputs = LIP_SYNCER_PURE_MOTION_SLIDER)
	LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX.change(update_lip_syncer_motion_smoothing, inputs = LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX)
	LIP_SYNCER_MOTION_MASK_MODE_DROPDOWN.change(update_lip_syncer_motion_mask_mode, inputs = LIP_SYNCER_MOTION_MASK_MODE_DROPDOWN)
	LIP_SYNCER_MASK_BLUR_SLIDER.release(update_lip_syncer_mask_blur, inputs = LIP_SYNCER_MASK_BLUR_SLIDER)
	LIP_SYNCER_MASK_ERODE_SLIDER.release(update_lip_syncer_mask_erode, inputs = LIP_SYNCER_MASK_ERODE_SLIDER)
	LIP_SYNCER_MASK_EXPAND_SLIDER.release(update_lip_syncer_mask_expand, inputs = LIP_SYNCER_MASK_EXPAND_SLIDER)
	LIP_SYNCER_CHIN_EXPAND_SLIDER.release(update_lip_syncer_chin_expand, inputs = LIP_SYNCER_CHIN_EXPAND_SLIDER)
	LIP_SYNCER_OCCLUSION_DILATE_SLIDER.release(update_lip_syncer_occlusion_dilate, inputs = LIP_SYNCER_OCCLUSION_DILATE_SLIDER)
	LIP_SYNCER_OCCLUSION_BLUR_SLIDER.release(update_lip_syncer_occlusion_blur, inputs = LIP_SYNCER_OCCLUSION_BLUR_SLIDER)
	mask_padding_sliders = [ LIP_SYNCER_MASK_PADDING_TOP_SLIDER, LIP_SYNCER_MASK_PADDING_RIGHT_SLIDER, LIP_SYNCER_MASK_PADDING_BOTTOM_SLIDER, LIP_SYNCER_MASK_PADDING_LEFT_SLIDER ]
	for mask_padding_slider in mask_padding_sliders:
		mask_padding_slider.release(update_face_mask_padding, inputs = mask_padding_sliders)
	LIP_SYNCER_EXPRESSIVENESS_SLIDER.release(update_lip_syncer_expressiveness, inputs = LIP_SYNCER_EXPRESSIVENESS_SLIDER)
	LIP_SYNCER_WEIGHT_SLIDER.release(update_lip_syncer_weight, inputs = LIP_SYNCER_WEIGHT_SLIDER)

	processors_checkbox_group = get_ui_component('processors_checkbox_group')
	if processors_checkbox_group:
		processors_checkbox_group.change(remote_update, inputs = processors_checkbox_group, outputs = [ LIP_SYNCER_MODEL_DROPDOWN, LIP_SYNCER_PURE_MOTION_SLIDER, LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX, LIP_SYNCER_MOTION_MASK_MODE_DROPDOWN, LIP_SYNCER_MASK_BLUR_SLIDER, LIP_SYNCER_MASK_ERODE_SLIDER, LIP_SYNCER_MASK_EXPAND_SLIDER, LIP_SYNCER_CHIN_EXPAND_SLIDER, LIP_SYNCER_OCCLUSION_DILATE_SLIDER, LIP_SYNCER_OCCLUSION_BLUR_SLIDER, LIP_SYNCER_MASK_PADDING_TOP_SLIDER, LIP_SYNCER_MASK_PADDING_RIGHT_SLIDER, LIP_SYNCER_MASK_PADDING_BOTTOM_SLIDER, LIP_SYNCER_MASK_PADDING_LEFT_SLIDER, LIP_SYNCER_EXPRESSIVENESS_SLIDER, LIP_SYNCER_WEIGHT_SLIDER ])


def remote_update(processors : List[str]) -> Tuple[gradio.Dropdown, gradio.Slider, gradio.Checkbox, gradio.Dropdown, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider]:
	has_lip_syncer = 'lip_syncer' in processors
	return gradio.Dropdown(visible = has_lip_syncer), gradio.Slider(visible = has_lip_syncer), gradio.Checkbox(visible = has_lip_syncer), gradio.Dropdown(visible = has_lip_syncer), gradio.Slider(visible = has_lip_syncer), gradio.Slider(visible = has_lip_syncer), gradio.Slider(visible = has_lip_syncer), gradio.Slider(visible = has_lip_syncer), gradio.Slider(visible = has_lip_syncer), gradio.Slider(visible = has_lip_syncer), gradio.Slider(visible = has_lip_syncer), gradio.Slider(visible = has_lip_syncer), gradio.Slider(visible = has_lip_syncer), gradio.Slider(visible = has_lip_syncer), gradio.Slider(visible = has_lip_syncer), gradio.Slider(visible = has_lip_syncer)


def update_lip_syncer_model(lip_syncer_model : LipSyncerModel) -> gradio.Dropdown:
	lip_syncer_module = load_processor_module('lip_syncer')
	lip_syncer_module.clear_inference_pool()
	state_manager.set_item('lip_syncer_model', lip_syncer_model)

	if lip_syncer_module.pre_check():
		return gradio.Dropdown(value = state_manager.get_item('lip_syncer_model'))
	return gradio.Dropdown()


def update_lip_syncer_pure_motion(lip_syncer_pure_motion : float) -> None:
	lip_syncer_module = load_processor_module('lip_syncer')
	lip_syncer_module.clear_inference_pool()
	state_manager.set_item('lip_syncer_pure_motion', lip_syncer_pure_motion)


def update_lip_syncer_motion_smoothing(lip_syncer_motion_smoothing : bool) -> None:
	lip_syncer_module = load_processor_module('lip_syncer')
	lip_syncer_module.clear_driver_expression_cache()
	state_manager.set_item('lip_syncer_motion_smoothing', lip_syncer_motion_smoothing)


def update_lip_syncer_motion_mask_mode(lip_syncer_motion_mask_mode : LipSyncerMotionMaskMode) -> None:
	state_manager.set_item('lip_syncer_motion_mask_mode', lip_syncer_motion_mask_mode)


def update_lip_syncer_mask_blur(lip_syncer_mask_blur : float) -> None:
	state_manager.set_item('lip_syncer_mask_blur', lip_syncer_mask_blur)


def update_lip_syncer_mask_erode(lip_syncer_mask_erode : int) -> None:
	state_manager.set_item('lip_syncer_mask_erode', int(lip_syncer_mask_erode))


def update_lip_syncer_mask_expand(lip_syncer_mask_expand : int) -> None:
	state_manager.set_item('lip_syncer_mask_expand', int(lip_syncer_mask_expand))


def update_lip_syncer_chin_expand(lip_syncer_chin_expand : int) -> None:
	state_manager.set_item('lip_syncer_chin_expand', int(lip_syncer_chin_expand))


def update_lip_syncer_occlusion_dilate(lip_syncer_occlusion_dilate : int) -> None:
	state_manager.set_item('lip_syncer_occlusion_dilate', int(lip_syncer_occlusion_dilate))


def update_lip_syncer_occlusion_blur(lip_syncer_occlusion_blur : float) -> None:
	state_manager.set_item('lip_syncer_occlusion_blur', lip_syncer_occlusion_blur)


def update_face_mask_padding(face_mask_padding_top : float, face_mask_padding_right : float, face_mask_padding_bottom : float, face_mask_padding_left : float) -> None:
	face_mask_padding_top = sanitize_int_range(int(face_mask_padding_top), facefusion.choices.face_mask_padding_range)
	face_mask_padding_right = sanitize_int_range(int(face_mask_padding_right), facefusion.choices.face_mask_padding_range)
	face_mask_padding_bottom = sanitize_int_range(int(face_mask_padding_bottom), facefusion.choices.face_mask_padding_range)
	face_mask_padding_left = sanitize_int_range(int(face_mask_padding_left), facefusion.choices.face_mask_padding_range)
	state_manager.set_item('face_mask_padding', (face_mask_padding_top, face_mask_padding_right, face_mask_padding_bottom, face_mask_padding_left))


def update_lip_syncer_expressiveness(lip_syncer_expressiveness : float) -> None:
	lip_syncer_module = load_processor_module('lip_syncer')
	lip_syncer_module.clear_driver_expression_cache()
	state_manager.set_item('lip_syncer_expressiveness', lip_syncer_expressiveness)


def update_lip_syncer_weight(lip_syncer_weight : LipSyncerWeight) -> None:
	lip_syncer_module = load_processor_module('lip_syncer')
	lip_syncer_module.clear_driver_expression_cache()
	state_manager.set_item('lip_syncer_weight', lip_syncer_weight)
