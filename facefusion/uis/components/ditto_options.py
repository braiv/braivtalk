from typing import List, Optional, Tuple

import gradio

from facefusion import state_manager
from facefusion.processors.modules.ditto import choices as ditto_choices
from facefusion.processors.modules.ditto.locales import get as locale_get
from facefusion.processors.modules.ditto.types import DittoBackend, DittoCompositeGeometryMode, DittoCropPrepMode, DittoRenderMode, DittoSourceMode
from facefusion.uis.core import get_ui_component, register_ui_component

DITTO_SOURCE_MODE_DROPDOWN : Optional[gradio.Dropdown] = None
DITTO_BACKEND_DROPDOWN : Optional[gradio.Dropdown] = None
DITTO_RENDER_MODE_DROPDOWN : Optional[gradio.Dropdown] = None
DITTO_REGISTRATION_CROP_MODE_DROPDOWN : Optional[gradio.Dropdown] = None
DITTO_LIVE_CROP_MODE_DROPDOWN : Optional[gradio.Dropdown] = None
DITTO_COMPOSITE_GEOMETRY_MODE_DROPDOWN : Optional[gradio.Dropdown] = None


def render() -> None:
	global DITTO_SOURCE_MODE_DROPDOWN
	global DITTO_BACKEND_DROPDOWN
	global DITTO_RENDER_MODE_DROPDOWN
	global DITTO_REGISTRATION_CROP_MODE_DROPDOWN
	global DITTO_LIVE_CROP_MODE_DROPDOWN
	global DITTO_COMPOSITE_GEOMETRY_MODE_DROPDOWN

	has_ditto = 'lip_syncer' in state_manager.get_item('processors') and state_manager.get_item('lip_syncer_pipeline') == 'ditto'
	DITTO_SOURCE_MODE_DROPDOWN = gradio.Dropdown(
		label = locale_get('uis.source_mode_dropdown'),
		choices = ditto_choices.ditto_source_modes,
		value = state_manager.get_item('ditto_source_mode'),
		visible = has_ditto
	)
	DITTO_BACKEND_DROPDOWN = gradio.Dropdown(
		label = locale_get('uis.backend_dropdown'),
		choices = ditto_choices.ditto_backends,
		value = state_manager.get_item('ditto_backend'),
		visible = has_ditto
	)
	DITTO_RENDER_MODE_DROPDOWN = gradio.Dropdown(
		label = locale_get('uis.render_mode_dropdown'),
		choices = ditto_choices.ditto_render_modes,
		value = state_manager.get_item('ditto_render_mode'),
		visible = has_ditto
	)
	DITTO_REGISTRATION_CROP_MODE_DROPDOWN = gradio.Dropdown(
		label = locale_get('uis.registration_crop_mode_dropdown'),
		choices = ditto_choices.ditto_crop_prep_modes,
		value = state_manager.get_item('ditto_registration_crop_mode'),
		visible = has_ditto
	)
	DITTO_LIVE_CROP_MODE_DROPDOWN = gradio.Dropdown(
		label = locale_get('uis.live_crop_mode_dropdown'),
		choices = ditto_choices.ditto_crop_prep_modes,
		value = state_manager.get_item('ditto_live_crop_mode'),
		visible = has_ditto
	)
	DITTO_COMPOSITE_GEOMETRY_MODE_DROPDOWN = gradio.Dropdown(
		label = locale_get('uis.composite_geometry_mode_dropdown'),
		choices = ditto_choices.ditto_composite_geometry_modes,
		value = state_manager.get_item('ditto_composite_geometry_mode'),
		visible = has_ditto
	)
	register_ui_component('ditto_source_mode_dropdown', DITTO_SOURCE_MODE_DROPDOWN)
	register_ui_component('ditto_backend_dropdown', DITTO_BACKEND_DROPDOWN)
	register_ui_component('ditto_render_mode_dropdown', DITTO_RENDER_MODE_DROPDOWN)
	register_ui_component('ditto_registration_crop_mode_dropdown', DITTO_REGISTRATION_CROP_MODE_DROPDOWN)
	register_ui_component('ditto_live_crop_mode_dropdown', DITTO_LIVE_CROP_MODE_DROPDOWN)
	register_ui_component('ditto_composite_geometry_mode_dropdown', DITTO_COMPOSITE_GEOMETRY_MODE_DROPDOWN)


def listen() -> None:
	DITTO_SOURCE_MODE_DROPDOWN.change(update_ditto_source_mode, inputs = DITTO_SOURCE_MODE_DROPDOWN)
	DITTO_BACKEND_DROPDOWN.change(update_ditto_backend, inputs = DITTO_BACKEND_DROPDOWN)
	DITTO_RENDER_MODE_DROPDOWN.change(update_ditto_render_mode, inputs = DITTO_RENDER_MODE_DROPDOWN)
	DITTO_REGISTRATION_CROP_MODE_DROPDOWN.change(update_ditto_registration_crop_mode, inputs = DITTO_REGISTRATION_CROP_MODE_DROPDOWN)
	DITTO_LIVE_CROP_MODE_DROPDOWN.change(update_ditto_live_crop_mode, inputs = DITTO_LIVE_CROP_MODE_DROPDOWN)
	DITTO_COMPOSITE_GEOMETRY_MODE_DROPDOWN.change(update_ditto_composite_geometry_mode, inputs = DITTO_COMPOSITE_GEOMETRY_MODE_DROPDOWN)

	processors_checkbox_group = get_ui_component('processors_checkbox_group')
	lip_syncer_pipeline_dropdown = get_ui_component('lip_syncer_pipeline_dropdown')
	if processors_checkbox_group:
		processors_checkbox_group.change(remote_update, inputs = processors_checkbox_group, outputs = [ DITTO_SOURCE_MODE_DROPDOWN, DITTO_BACKEND_DROPDOWN, DITTO_RENDER_MODE_DROPDOWN, DITTO_REGISTRATION_CROP_MODE_DROPDOWN, DITTO_LIVE_CROP_MODE_DROPDOWN, DITTO_COMPOSITE_GEOMETRY_MODE_DROPDOWN ])
	if lip_syncer_pipeline_dropdown:
		lip_syncer_pipeline_dropdown.change(remote_update_for_pipeline, inputs = lip_syncer_pipeline_dropdown, outputs = [ DITTO_SOURCE_MODE_DROPDOWN, DITTO_BACKEND_DROPDOWN, DITTO_RENDER_MODE_DROPDOWN, DITTO_REGISTRATION_CROP_MODE_DROPDOWN, DITTO_LIVE_CROP_MODE_DROPDOWN, DITTO_COMPOSITE_GEOMETRY_MODE_DROPDOWN ])


def remote_update(processors : List[str]) -> Tuple[gradio.Dropdown, gradio.Dropdown, gradio.Dropdown, gradio.Dropdown, gradio.Dropdown, gradio.Dropdown]:
	has_ditto = 'lip_syncer' in processors and state_manager.get_item('lip_syncer_pipeline') == 'ditto'
	return gradio.Dropdown(visible = has_ditto), gradio.Dropdown(visible = has_ditto), gradio.Dropdown(visible = has_ditto), gradio.Dropdown(visible = has_ditto), gradio.Dropdown(visible = has_ditto), gradio.Dropdown(visible = has_ditto)


def remote_update_for_pipeline(lip_syncer_pipeline : str) -> Tuple[gradio.Dropdown, gradio.Dropdown, gradio.Dropdown, gradio.Dropdown, gradio.Dropdown, gradio.Dropdown]:
	has_ditto = 'lip_syncer' in state_manager.get_item('processors') and lip_syncer_pipeline == 'ditto'
	return gradio.Dropdown(visible = has_ditto), gradio.Dropdown(visible = has_ditto), gradio.Dropdown(visible = has_ditto), gradio.Dropdown(visible = has_ditto), gradio.Dropdown(visible = has_ditto), gradio.Dropdown(visible = has_ditto)


def update_ditto_source_mode(ditto_source_mode : DittoSourceMode) -> None:
	state_manager.set_item('ditto_source_mode', ditto_source_mode)


def update_ditto_backend(ditto_backend : DittoBackend) -> None:
	state_manager.set_item('ditto_backend', ditto_backend)


def update_ditto_render_mode(ditto_render_mode : DittoRenderMode) -> None:
	state_manager.set_item('ditto_render_mode', ditto_render_mode)


def update_ditto_registration_crop_mode(ditto_registration_crop_mode : DittoCropPrepMode) -> None:
	state_manager.set_item('ditto_registration_crop_mode', ditto_registration_crop_mode)


def update_ditto_live_crop_mode(ditto_live_crop_mode : DittoCropPrepMode) -> None:
	state_manager.set_item('ditto_live_crop_mode', ditto_live_crop_mode)


def update_ditto_composite_geometry_mode(ditto_composite_geometry_mode : DittoCompositeGeometryMode) -> None:
	state_manager.set_item('ditto_composite_geometry_mode', ditto_composite_geometry_mode)
