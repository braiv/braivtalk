from typing import Optional

import gradio

import facefusion.choices
from facefusion import face_masker, state_manager, translator
from facefusion.common_helper import calculate_float_step
from facefusion.types import FaceOccluderModel, FaceParserModel
from facefusion.uis.core import register_ui_component

FACE_OCCLUDER_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
FACE_PARSER_MODEL_DROPDOWN : Optional[gradio.Dropdown] = None
FACE_MASK_OCCLUSION_CHECKBOX : Optional[gradio.Checkbox] = None
FACE_MASK_BLUR_SLIDER : Optional[gradio.Slider] = None


def render() -> None:
	global FACE_OCCLUDER_MODEL_DROPDOWN
	global FACE_PARSER_MODEL_DROPDOWN
	global FACE_MASK_OCCLUSION_CHECKBOX
	global FACE_MASK_BLUR_SLIDER

	has_occlusion = 'occlusion' in state_manager.get_item('face_mask_types')
	with gradio.Row():
		FACE_OCCLUDER_MODEL_DROPDOWN = gradio.Dropdown(
			label = translator.get('uis.face_occluder_model_dropdown'),
			choices = facefusion.choices.face_occluder_models,
			value = state_manager.get_item('face_occluder_model')
		)
		FACE_PARSER_MODEL_DROPDOWN = gradio.Dropdown(
			label = translator.get('uis.face_parser_model_dropdown'),
			choices = facefusion.choices.face_parser_models,
			value = state_manager.get_item('face_parser_model')
		)
	FACE_MASK_OCCLUSION_CHECKBOX = gradio.Checkbox(
		label = translator.get('uis.face_mask_occlusion_checkbox'),
		value = has_occlusion
	)
	FACE_MASK_BLUR_SLIDER = gradio.Slider(
		label = translator.get('uis.face_mask_blur_slider'),
		step = calculate_float_step(facefusion.choices.face_mask_blur_range),
		minimum = facefusion.choices.face_mask_blur_range[0],
		maximum = facefusion.choices.face_mask_blur_range[-1],
		value = state_manager.get_item('face_mask_blur')
	)
	register_ui_component('face_occluder_model_dropdown', FACE_OCCLUDER_MODEL_DROPDOWN)
	register_ui_component('face_parser_model_dropdown', FACE_PARSER_MODEL_DROPDOWN)
	register_ui_component('face_mask_occlusion_checkbox', FACE_MASK_OCCLUSION_CHECKBOX)
	register_ui_component('face_mask_blur_slider', FACE_MASK_BLUR_SLIDER)


def listen() -> None:
	FACE_OCCLUDER_MODEL_DROPDOWN.change(update_face_occluder_model, inputs = FACE_OCCLUDER_MODEL_DROPDOWN)
	FACE_PARSER_MODEL_DROPDOWN.change(update_face_parser_model, inputs = FACE_PARSER_MODEL_DROPDOWN)
	FACE_MASK_OCCLUSION_CHECKBOX.change(update_face_mask_occlusion, inputs = FACE_MASK_OCCLUSION_CHECKBOX)
	FACE_MASK_BLUR_SLIDER.release(update_face_mask_blur, inputs = FACE_MASK_BLUR_SLIDER)


def update_face_occluder_model(face_occluder_model : FaceOccluderModel) -> gradio.Dropdown:
	face_masker.clear_inference_pool()
	state_manager.set_item('face_occluder_model', face_occluder_model)

	if face_masker.pre_check():
		return gradio.Dropdown(value = state_manager.get_item('face_occluder_model'))
	return gradio.Dropdown()


def update_face_parser_model(face_parser_model : FaceParserModel) -> gradio.Dropdown:
	face_masker.clear_inference_pool()
	state_manager.set_item('face_parser_model', face_parser_model)

	if face_masker.pre_check():
		return gradio.Dropdown(value = state_manager.get_item('face_parser_model'))
	return gradio.Dropdown()


def update_face_mask_occlusion(occlusion_enabled : bool) -> None:
	face_mask_types = list(state_manager.get_item('face_mask_types') or [])
	if occlusion_enabled and 'occlusion' not in face_mask_types:
		face_mask_types.append('occlusion')
	elif not occlusion_enabled and 'occlusion' in face_mask_types:
		face_mask_types.remove('occlusion')
	state_manager.set_item('face_mask_types', face_mask_types)


def update_face_mask_blur(face_mask_blur : float) -> None:
	state_manager.set_item('face_mask_blur', face_mask_blur)
