from typing import List, Optional, Tuple

import gradio

from facefusion import state_manager, translator
from facefusion.common_helper import calculate_float_step, calculate_int_step
from facefusion.processors.core import load_processor_module
from facefusion.processors.modules.lip_syncer import choices as lip_syncer_choices
from facefusion.processors.modules.lip_syncer.types import LipSyncerPipeline, LipSyncerWeight
from facefusion.uis.core import get_ui_component, register_ui_component

LIP_SYNCER_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
LIP_SYNCER_PURE_MOTION_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_MOTION_DAMPING_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_CROP_STABILIZATION_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX : Optional[gradio.Checkbox] = None
LIP_SYNCER_MOTION_MASK_MODE_DROPDOWN : Optional[gradio.Dropdown] = None
LIP_SYNCER_MASK_BLUR_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_MASK_ERODE_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_MASK_EXPAND_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_CHIN_EXPAND_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_OCCLUSION_DILATE_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_OCCLUSION_BLUR_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_EXPRESSIVENESS_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_WEIGHT_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global LIP_SYNCER_MODEL_DROPDOWN
	global LIP_SYNCER_PURE_MOTION_SLIDER
	global LIP_SYNCER_MOTION_DAMPING_SLIDER
	global LIP_SYNCER_CROP_STABILIZATION_SLIDER
	global LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX
	global LIP_SYNCER_MOTION_MASK_MODE_DROPDOWN
	global LIP_SYNCER_MASK_BLUR_SLIDER
	global LIP_SYNCER_MASK_ERODE_SLIDER
	global LIP_SYNCER_MASK_EXPAND_SLIDER
	global LIP_SYNCER_CHIN_EXPAND_SLIDER
	global LIP_SYNCER_OCCLUSION_DILATE_SLIDER
	global LIP_SYNCER_OCCLUSION_BLUR_SLIDER
	global LIP_SYNCER_EXPRESSIVENESS_SLIDER
	global LIP_SYNCER_WEIGHT_SLIDER

	has_lip_syncer = 'lip_syncer' in state_manager.get_item('processors')
	has_ditto = 'ditto' in state_manager.get_item('processors')
	has_shared_live_portrait_controls = has_lip_syncer or has_ditto
	is_live_portrait = state_manager.get_item('lip_syncer_pipeline') == 'live_portrait'
	LIP_SYNCER_MODEL_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.pipeline_dropdown', 'facefusion.processors.modules.lip_syncer'),
		choices = lip_syncer_choices.lip_syncer_pipelines,
		value = state_manager.get_item('lip_syncer_pipeline'),
		visible = has_lip_syncer
	)
	LIP_SYNCER_PURE_MOTION_SLIDER = gradio.Slider(
		label = translator.get('uis.pure_motion_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_pure_motion'),
		step = calculate_float_step(lip_syncer_choices.lip_syncer_pure_motion_range),
		minimum = lip_syncer_choices.lip_syncer_pure_motion_range[0],
		maximum = lip_syncer_choices.lip_syncer_pure_motion_range[-1],
		visible = has_lip_syncer and is_live_portrait
	)
	LIP_SYNCER_MOTION_DAMPING_SLIDER = gradio.Slider(
		label = translator.get('uis.motion_damping_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_motion_damping'),
		step = calculate_float_step(lip_syncer_choices.lip_syncer_motion_damping_range),
		minimum = lip_syncer_choices.lip_syncer_motion_damping_range[0],
		maximum = lip_syncer_choices.lip_syncer_motion_damping_range[-1],
		visible = has_lip_syncer and is_live_portrait
	)
	LIP_SYNCER_CROP_STABILIZATION_SLIDER = gradio.Slider(
		label = translator.get('uis.crop_stabilization_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_crop_stabilization'),
		step = calculate_float_step(lip_syncer_choices.lip_syncer_crop_stabilization_range),
		minimum = lip_syncer_choices.lip_syncer_crop_stabilization_range[0],
		maximum = lip_syncer_choices.lip_syncer_crop_stabilization_range[-1],
		visible = has_shared_live_portrait_controls
	)
	LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX = gradio.Checkbox(
		label = translator.get('uis.motion_smoothing_checkbox', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_motion_smoothing'),
		visible = has_lip_syncer and is_live_portrait
	)
	LIP_SYNCER_MOTION_MASK_MODE_DROPDOWN = gradio.Dropdown(
		label = translator.get('uis.motion_mask_mode_dropdown', 'facefusion.processors.modules.lip_syncer'),
		choices = lip_syncer_choices.lip_syncer_motion_mask_modes,
		value = state_manager.get_item('lip_syncer_motion_mask_mode'),
		visible = has_shared_live_portrait_controls
	)
	LIP_SYNCER_MASK_BLUR_SLIDER = gradio.Slider(
		label = translator.get('uis.mask_blur_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_mask_blur'),
		step = calculate_float_step(lip_syncer_choices.lip_syncer_mask_blur_range),
		minimum = lip_syncer_choices.lip_syncer_mask_blur_range[0],
		maximum = lip_syncer_choices.lip_syncer_mask_blur_range[-1],
		visible = has_shared_live_portrait_controls
	)
	LIP_SYNCER_MASK_ERODE_SLIDER = gradio.Slider(
		label = translator.get('uis.mask_erode_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_mask_erode'),
		step = calculate_int_step(lip_syncer_choices.lip_syncer_mask_erode_range),
		minimum = lip_syncer_choices.lip_syncer_mask_erode_range[0],
		maximum = lip_syncer_choices.lip_syncer_mask_erode_range[-1],
		visible = has_shared_live_portrait_controls
	)
	LIP_SYNCER_MASK_EXPAND_SLIDER = gradio.Slider(
		label = translator.get('uis.mask_expand_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_mask_expand'),
		step = calculate_int_step(lip_syncer_choices.lip_syncer_mask_expand_range),
		minimum = lip_syncer_choices.lip_syncer_mask_expand_range[0],
		maximum = lip_syncer_choices.lip_syncer_mask_expand_range[-1],
		visible = has_shared_live_portrait_controls
	)
	LIP_SYNCER_CHIN_EXPAND_SLIDER = gradio.Slider(
		label = translator.get('uis.chin_expand_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_chin_expand'),
		step = calculate_int_step(lip_syncer_choices.lip_syncer_chin_expand_range),
		minimum = lip_syncer_choices.lip_syncer_chin_expand_range[0],
		maximum = lip_syncer_choices.lip_syncer_chin_expand_range[-1],
		visible = has_shared_live_portrait_controls
	)
	LIP_SYNCER_OCCLUSION_DILATE_SLIDER = gradio.Slider(
		label = translator.get('uis.occlusion_dilate_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_occlusion_dilate'),
		step = calculate_int_step(lip_syncer_choices.lip_syncer_occlusion_dilate_range),
		minimum = lip_syncer_choices.lip_syncer_occlusion_dilate_range[0],
		maximum = lip_syncer_choices.lip_syncer_occlusion_dilate_range[-1],
		visible = has_shared_live_portrait_controls
	)
	LIP_SYNCER_OCCLUSION_BLUR_SLIDER = gradio.Slider(
		label = translator.get('uis.occlusion_blur_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_occlusion_blur'),
		step = calculate_float_step(lip_syncer_choices.lip_syncer_occlusion_blur_range),
		minimum = lip_syncer_choices.lip_syncer_occlusion_blur_range[0],
		maximum = lip_syncer_choices.lip_syncer_occlusion_blur_range[-1],
		visible = has_shared_live_portrait_controls
	)
	LIP_SYNCER_EXPRESSIVENESS_SLIDER = gradio.Slider(
		label = translator.get('uis.expressiveness_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_expressiveness'),
		step = calculate_float_step(lip_syncer_choices.lip_syncer_expressiveness_range),
		minimum = lip_syncer_choices.lip_syncer_expressiveness_range[0],
		maximum = lip_syncer_choices.lip_syncer_expressiveness_range[-1],
		visible = has_shared_live_portrait_controls
	)
	LIP_SYNCER_WEIGHT_SLIDER = gradio.Slider(
		label = translator.get('uis.weight_slider', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_weight'),
		step = calculate_float_step(lip_syncer_choices.lip_syncer_weight_range),
		minimum = lip_syncer_choices.lip_syncer_weight_range[0],
		maximum = lip_syncer_choices.lip_syncer_weight_range[-1],
		visible = has_lip_syncer and is_live_portrait
	)
	register_ui_component('lip_syncer_model_dropdown', LIP_SYNCER_MODEL_DROPDOWN)
	register_ui_component('lip_syncer_pipeline_dropdown', LIP_SYNCER_MODEL_DROPDOWN)
	register_ui_component('lip_syncer_pure_motion_slider', LIP_SYNCER_PURE_MOTION_SLIDER)
	register_ui_component('lip_syncer_motion_damping_slider', LIP_SYNCER_MOTION_DAMPING_SLIDER)
	register_ui_component('lip_syncer_crop_stabilization_slider', LIP_SYNCER_CROP_STABILIZATION_SLIDER)
	register_ui_component('lip_syncer_motion_smoothing_checkbox', LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX)
	register_ui_component('lip_syncer_motion_mask_mode_dropdown', LIP_SYNCER_MOTION_MASK_MODE_DROPDOWN)
	register_ui_component('lip_syncer_mask_blur_slider', LIP_SYNCER_MASK_BLUR_SLIDER)
	register_ui_component('lip_syncer_mask_erode_slider', LIP_SYNCER_MASK_ERODE_SLIDER)
	register_ui_component('lip_syncer_mask_expand_slider', LIP_SYNCER_MASK_EXPAND_SLIDER)
	register_ui_component('lip_syncer_chin_expand_slider', LIP_SYNCER_CHIN_EXPAND_SLIDER)
	register_ui_component('lip_syncer_occlusion_dilate_slider', LIP_SYNCER_OCCLUSION_DILATE_SLIDER)
	register_ui_component('lip_syncer_occlusion_blur_slider', LIP_SYNCER_OCCLUSION_BLUR_SLIDER)
	register_ui_component('lip_syncer_expressiveness_slider', LIP_SYNCER_EXPRESSIVENESS_SLIDER)
	register_ui_component('lip_syncer_weight_slider', LIP_SYNCER_WEIGHT_SLIDER)


def listen() -> None:
	LIP_SYNCER_MODEL_DROPDOWN.change(update_lip_syncer_pipeline, inputs = LIP_SYNCER_MODEL_DROPDOWN, outputs = [ LIP_SYNCER_MODEL_DROPDOWN, LIP_SYNCER_PURE_MOTION_SLIDER, LIP_SYNCER_MOTION_DAMPING_SLIDER, LIP_SYNCER_CROP_STABILIZATION_SLIDER, LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX, LIP_SYNCER_MOTION_MASK_MODE_DROPDOWN, LIP_SYNCER_MASK_BLUR_SLIDER, LIP_SYNCER_MASK_ERODE_SLIDER, LIP_SYNCER_MASK_EXPAND_SLIDER, LIP_SYNCER_CHIN_EXPAND_SLIDER, LIP_SYNCER_OCCLUSION_DILATE_SLIDER, LIP_SYNCER_OCCLUSION_BLUR_SLIDER, LIP_SYNCER_EXPRESSIVENESS_SLIDER, LIP_SYNCER_WEIGHT_SLIDER ])
	LIP_SYNCER_PURE_MOTION_SLIDER.release(update_lip_syncer_pure_motion, inputs = LIP_SYNCER_PURE_MOTION_SLIDER)
	LIP_SYNCER_MOTION_DAMPING_SLIDER.release(update_lip_syncer_motion_damping, inputs = LIP_SYNCER_MOTION_DAMPING_SLIDER)
	LIP_SYNCER_CROP_STABILIZATION_SLIDER.release(update_lip_syncer_crop_stabilization, inputs = LIP_SYNCER_CROP_STABILIZATION_SLIDER)
	LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX.change(update_lip_syncer_motion_smoothing, inputs = LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX)
	LIP_SYNCER_MOTION_MASK_MODE_DROPDOWN.change(update_lip_syncer_motion_mask_mode, inputs = LIP_SYNCER_MOTION_MASK_MODE_DROPDOWN)
	LIP_SYNCER_MASK_BLUR_SLIDER.release(update_lip_syncer_mask_blur, inputs = LIP_SYNCER_MASK_BLUR_SLIDER)
	LIP_SYNCER_MASK_ERODE_SLIDER.release(update_lip_syncer_mask_erode, inputs = LIP_SYNCER_MASK_ERODE_SLIDER)
	LIP_SYNCER_MASK_EXPAND_SLIDER.release(update_lip_syncer_mask_expand, inputs = LIP_SYNCER_MASK_EXPAND_SLIDER)
	LIP_SYNCER_CHIN_EXPAND_SLIDER.release(update_lip_syncer_chin_expand, inputs = LIP_SYNCER_CHIN_EXPAND_SLIDER)
	LIP_SYNCER_OCCLUSION_DILATE_SLIDER.release(update_lip_syncer_occlusion_dilate, inputs = LIP_SYNCER_OCCLUSION_DILATE_SLIDER)
	LIP_SYNCER_OCCLUSION_BLUR_SLIDER.release(update_lip_syncer_occlusion_blur, inputs = LIP_SYNCER_OCCLUSION_BLUR_SLIDER)
	LIP_SYNCER_EXPRESSIVENESS_SLIDER.release(update_lip_syncer_expressiveness, inputs = LIP_SYNCER_EXPRESSIVENESS_SLIDER)
	LIP_SYNCER_WEIGHT_SLIDER.release(update_lip_syncer_weight, inputs = LIP_SYNCER_WEIGHT_SLIDER)

	processors_checkbox_group = get_ui_component('processors_checkbox_group')
	if processors_checkbox_group:
		processors_checkbox_group.change(remote_update, inputs = processors_checkbox_group, outputs = [ LIP_SYNCER_MODEL_DROPDOWN, LIP_SYNCER_PURE_MOTION_SLIDER, LIP_SYNCER_MOTION_DAMPING_SLIDER, LIP_SYNCER_CROP_STABILIZATION_SLIDER, LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX, LIP_SYNCER_MOTION_MASK_MODE_DROPDOWN, LIP_SYNCER_MASK_BLUR_SLIDER, LIP_SYNCER_MASK_ERODE_SLIDER, LIP_SYNCER_MASK_EXPAND_SLIDER, LIP_SYNCER_CHIN_EXPAND_SLIDER, LIP_SYNCER_OCCLUSION_DILATE_SLIDER, LIP_SYNCER_OCCLUSION_BLUR_SLIDER, LIP_SYNCER_EXPRESSIVENESS_SLIDER, LIP_SYNCER_WEIGHT_SLIDER ])


def remote_update(processors : List[str]) -> Tuple[gradio.Dropdown, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Checkbox, gradio.Dropdown, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider]:
	has_lip_syncer = 'lip_syncer' in processors
	has_ditto = 'ditto' in processors
	has_shared_live_portrait_controls = has_lip_syncer or has_ditto
	is_live_portrait = state_manager.get_item('lip_syncer_pipeline') == 'live_portrait'
	return gradio.Dropdown(visible = has_lip_syncer), gradio.Slider(visible = has_lip_syncer and is_live_portrait), gradio.Slider(visible = has_lip_syncer and is_live_portrait), gradio.Slider(visible = has_shared_live_portrait_controls), gradio.Checkbox(visible = has_lip_syncer and is_live_portrait), gradio.Dropdown(visible = has_shared_live_portrait_controls), gradio.Slider(visible = has_shared_live_portrait_controls), gradio.Slider(visible = has_shared_live_portrait_controls), gradio.Slider(visible = has_shared_live_portrait_controls), gradio.Slider(visible = has_shared_live_portrait_controls), gradio.Slider(visible = has_shared_live_portrait_controls), gradio.Slider(visible = has_shared_live_portrait_controls), gradio.Slider(visible = has_shared_live_portrait_controls), gradio.Slider(visible = has_lip_syncer and is_live_portrait)


def update_lip_syncer_pipeline(lip_syncer_pipeline : LipSyncerPipeline) -> Tuple[gradio.Dropdown, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Checkbox, gradio.Dropdown, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider, gradio.Slider]:
	lip_syncer_module = load_processor_module('lip_syncer')
	lip_syncer_module.clear_inference_pool()
	state_manager.set_item('lip_syncer_pipeline', lip_syncer_pipeline)
	updates = remote_update(state_manager.get_item('processors'))
	return (
		gradio.Dropdown(value = state_manager.get_item('lip_syncer_pipeline'), visible = 'lip_syncer' in state_manager.get_item('processors')),
		*updates[1:]
	)


def update_lip_syncer_pure_motion(lip_syncer_pure_motion : float) -> None:
	lip_syncer_module = load_processor_module('lip_syncer')
	lip_syncer_module.clear_inference_pool()
	state_manager.set_item('lip_syncer_pure_motion', lip_syncer_pure_motion)


def update_lip_syncer_motion_damping(lip_syncer_motion_damping : float) -> None:
	state_manager.set_item('lip_syncer_motion_damping', lip_syncer_motion_damping)


def update_lip_syncer_crop_stabilization(lip_syncer_crop_stabilization : float) -> None:
	state_manager.set_item('lip_syncer_crop_stabilization', lip_syncer_crop_stabilization)


def update_lip_syncer_motion_smoothing(lip_syncer_motion_smoothing : bool) -> None:
	lip_syncer_module = load_processor_module('lip_syncer')
	lip_syncer_module.clear_driver_expression_cache()
	state_manager.set_item('lip_syncer_motion_smoothing', lip_syncer_motion_smoothing)


def update_lip_syncer_motion_mask_mode(lip_syncer_motion_mask_mode : str) -> None:
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


def update_lip_syncer_expressiveness(lip_syncer_expressiveness : float) -> None:
	lip_syncer_module = load_processor_module('lip_syncer')
	lip_syncer_module.clear_driver_expression_cache()
	state_manager.set_item('lip_syncer_expressiveness', lip_syncer_expressiveness)


def update_lip_syncer_weight(lip_syncer_weight : LipSyncerWeight) -> None:
	lip_syncer_module = load_processor_module('lip_syncer')
	lip_syncer_module.clear_driver_expression_cache()
	state_manager.set_item('lip_syncer_weight', lip_syncer_weight)
