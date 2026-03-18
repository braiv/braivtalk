from argparse import ArgumentParser

from facefusion.processors.core import get_available_processors
from facefusion.program import create_program
from facefusion.uis.components.processors import sort_processors


def find_action(program : ArgumentParser, dest : str):
	for action in program._actions:
		if action.dest == dest:
			return action

	for action in program._actions:
		if hasattr(action, 'choices') and action.choices:
			for sub_program in action.choices.values():
				sub_action = find_action(sub_program, dest)
				if sub_action:
					return sub_action
	return None


def test_available_processors() -> None:
	assert get_available_processors() == [ 'lip_syncer', 'face_enhancer' ]


def test_processors_argument_uses_supported_allowlist() -> None:
	program = create_program()
	processors_action = find_action(program, 'processors')

	assert processors_action is not None
	assert processors_action.choices == [ 'lip_syncer', 'face_enhancer' ]
	assert processors_action.default == [ 'lip_syncer', 'face_enhancer' ]


def test_ui_processor_sorting_uses_supported_allowlist() -> None:
	assert sort_processors([ 'face_enhancer', 'unknown_processor' ]) == [ 'face_enhancer', 'lip_syncer' ]
