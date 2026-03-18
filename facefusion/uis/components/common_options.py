from typing import List, Optional

import gradio

from facefusion import config, state_manager, translator
from facefusion.uis import choices as uis_choices

COMMON_OPTIONS_CHECKBOX_GROUP : Optional[gradio.Checkboxgroup] = None
SAVE_AS_DEFAULT_BUTTON : Optional[gradio.Button] = None
SAVE_AS_DEFAULT_STATUS_TEXTBOX : Optional[gradio.Textbox] = None


def render() -> None:
	global COMMON_OPTIONS_CHECKBOX_GROUP
	global SAVE_AS_DEFAULT_BUTTON
	global SAVE_AS_DEFAULT_STATUS_TEXTBOX

	common_options = []

	if state_manager.get_item('keep_temp'):
		common_options.append('keep-temp')

	COMMON_OPTIONS_CHECKBOX_GROUP = gradio.Checkboxgroup(
		label = translator.get('uis.common_options_checkbox_group'),
		choices = uis_choices.common_options,
		value = common_options
	)
	with gradio.Row():
		SAVE_AS_DEFAULT_BUTTON = gradio.Button(
			value = translator.get('uis.save_as_default_button'),
			size = 'sm'
		)
	SAVE_AS_DEFAULT_STATUS_TEXTBOX = gradio.Textbox(
		show_label = False,
		value = None,
		max_lines = 1,
		interactive = False,
		visible = False
	)


def listen() -> None:
	COMMON_OPTIONS_CHECKBOX_GROUP.change(update, inputs = COMMON_OPTIONS_CHECKBOX_GROUP)
	SAVE_AS_DEFAULT_BUTTON.click(save_as_default, outputs = SAVE_AS_DEFAULT_STATUS_TEXTBOX)


def update(common_options : List[str]) -> None:
	keep_temp = 'keep-temp' in common_options
	state_manager.set_item('keep_temp', keep_temp)


def save_as_default() -> gradio.Textbox:
	try:
		config_path = config.save_defaults()
		status = translator.get('uis.save_as_default_status').format(config_path = config_path)
	except Exception as exception:
		status = translator.get('uis.save_as_default_error').format(error = str(exception))
	return gradio.Textbox(value = status, visible = True)
