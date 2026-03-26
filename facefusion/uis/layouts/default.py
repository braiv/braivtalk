import gradio

from facefusion import state_manager
from facefusion.uis.components import about, common_options, ditto_options, download, execution, execution_thread_count, face_detector, face_enhancer_options, face_landmarker, face_masker, face_selector, instant_runner, job_manager, job_runner, lip_syncer_options, memory, output, output_options, preview, preview_options, processors, source, target, temp_frame, terminal, trim_frame, ui_workflow, voice_extractor


def pre_check() -> bool:
	return True


def render() -> gradio.Blocks:
	with gradio.Blocks() as layout:
		with gradio.Row():
			with gradio.Column(scale = 4):
				with gradio.Blocks():
					about.render()
				with gradio.Blocks():
					processors.render()
				with gradio.Blocks():
					face_enhancer_options.render()
				with gradio.Blocks():
					lip_syncer_options.render()
				with gradio.Blocks():
					ditto_options.render()
				with gradio.Blocks():
					voice_extractor.render()
				with gradio.Blocks():
					execution.render()
					execution_thread_count.render()
				with gradio.Blocks():
					download.render()
				with gradio.Blocks():
					memory.render()
				with gradio.Blocks():
					temp_frame.render()
				with gradio.Blocks():
					output_options.render()
			with gradio.Column(scale = 4):
				with gradio.Blocks():
					source.render()
				with gradio.Blocks():
					target.render()
				with gradio.Blocks():
					output.render()
				with gradio.Blocks():
					terminal.render()
				with gradio.Blocks():
					ui_workflow.render()
					instant_runner.render()
					job_runner.render()
					job_manager.render()
			with gradio.Column(scale = 7):
				with gradio.Blocks():
					preview.render()
					preview_options.render()
				with gradio.Blocks():
					trim_frame.render()
				with gradio.Blocks():
					face_selector.render()
				with gradio.Blocks():
					face_masker.render()
				with gradio.Blocks():
					face_detector.render()
				with gradio.Blocks():
					face_landmarker.render()
				with gradio.Blocks():
					common_options.render()
	return layout


def listen() -> None:
	processors.listen()
	face_enhancer_options.listen()
	lip_syncer_options.listen()
	ditto_options.listen()
	execution.listen()
	execution_thread_count.listen()
	download.listen()
	memory.listen()
	temp_frame.listen()
	output_options.listen()
	source.listen()
	target.listen()
	output.listen()
	instant_runner.listen()
	job_runner.listen()
	job_manager.listen()
	terminal.listen()
	preview.listen()
	preview_options.listen()
	trim_frame.listen()
	face_selector.listen()
	face_masker.listen()
	face_detector.listen()
	face_landmarker.listen()
	voice_extractor.listen()
	common_options.listen()


def run(ui : gradio.Blocks) -> None:
	ui.launch(favicon_path = 'facefusion.ico', inbrowser = state_manager.get_item('open_browser'))
