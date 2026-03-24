from typing import List, Optional, Tuple

import gradio

from facefusion import state_manager, translator
from facefusion.common_helper import calculate_float_step
from facefusion.processors.core import load_processor_module
from facefusion.processors.modules.lip_syncer import choices as lip_syncer_choices
from facefusion.processors.modules.lip_syncer.types import LipSyncerModel, LipSyncerWeight
from facefusion.uis.core import get_ui_component, register_ui_component

LIP_SYNCER_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
LIP_SYNCER_PURE_MOTION_SLIDER : Optional[gradio.Slider] = None
LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX : Optional[gradio.Checkbox] = None
LIP_SYNCER_MOTION_MASKING_CHECKBOX : Optional[gradio.Checkbox] = None
LIP_SYNCER_WEIGHT_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global LIP_SYNCER_MODEL_DROPDOWN
	global LIP_SYNCER_PURE_MOTION_SLIDER
	global LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX
	global LIP_SYNCER_MOTION_MASKING_CHECKBOX
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
	LIP_SYNCER_MOTION_MASKING_CHECKBOX = gradio.Checkbox(
		label = translator.get('uis.motion_masking_checkbox', 'facefusion.processors.modules.lip_syncer'),
		value = state_manager.get_item('lip_syncer_motion_masking'),
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
	register_ui_component('lip_syncer_motion_masking_checkbox', LIP_SYNCER_MOTION_MASKING_CHECKBOX)
	register_ui_component('lip_syncer_weight_slider', LIP_SYNCER_WEIGHT_SLIDER)


def listen() -> None:
	LIP_SYNCER_MODEL_DROPDOWN.change(update_lip_syncer_model, inputs = LIP_SYNCER_MODEL_DROPDOWN, outputs = LIP_SYNCER_MODEL_DROPDOWN)
	LIP_SYNCER_PURE_MOTION_SLIDER.release(update_lip_syncer_pure_motion, inputs = LIP_SYNCER_PURE_MOTION_SLIDER)
	LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX.change(update_lip_syncer_motion_smoothing, inputs = LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX)
	LIP_SYNCER_MOTION_MASKING_CHECKBOX.change(update_lip_syncer_motion_masking, inputs = LIP_SYNCER_MOTION_MASKING_CHECKBOX)
	LIP_SYNCER_WEIGHT_SLIDER.release(update_lip_syncer_weight, inputs = LIP_SYNCER_WEIGHT_SLIDER)

	processors_checkbox_group = get_ui_component('processors_checkbox_group')
	if processors_checkbox_group:
		processors_checkbox_group.change(remote_update, inputs = processors_checkbox_group, outputs = [ LIP_SYNCER_MODEL_DROPDOWN, LIP_SYNCER_PURE_MOTION_SLIDER, LIP_SYNCER_MOTION_SMOOTHING_CHECKBOX, LIP_SYNCER_MOTION_MASKING_CHECKBOX, LIP_SYNCER_WEIGHT_SLIDER ])


def remote_update(processors : List[str]) -> Tuple[gradio.Dropdown, gradio.Slider, gradio.Checkbox, gradio.Checkbox, gradio.Slider]:
	has_lip_syncer = 'lip_syncer' in processors
	return gradio.Dropdown(visible = has_lip_syncer), gradio.Slider(visible = has_lip_syncer), gradio.Checkbox(visible = has_lip_syncer), gradio.Checkbox(visible = has_lip_syncer), gradio.Slider(visible = has_lip_syncer)


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


def update_lip_syncer_motion_masking(lip_syncer_motion_masking : bool) -> None:
	state_manager.set_item('lip_syncer_motion_masking', lip_syncer_motion_masking)


def update_lip_syncer_weight(lip_syncer_weight : LipSyncerWeight) -> None:
	lip_syncer_module = load_processor_module('lip_syncer')
	lip_syncer_module.clear_driver_expression_cache()
	state_manager.set_item('lip_syncer_weight', lip_syncer_weight)
