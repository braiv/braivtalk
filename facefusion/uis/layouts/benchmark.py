import gradio

from facefusion import benchmarker, state_manager
from facefusion.uis.components import about, benchmark, benchmark_options, download, execution, execution_thread_count, face_enhancer_options, lip_syncer_options, memory, processors


def pre_check() -> bool:
	return benchmarker.pre_check()


def render() -> gradio.Blocks:
	with gradio.Blocks() as layout:
		with gradio.Row():
			with gradio.Column(scale = 4):
				with gradio.Blocks():
					about.render()
				with gradio.Blocks():
					benchmark_options.render()
				with gradio.Blocks():
					processors.render()
				with gradio.Blocks():
					face_enhancer_options.render()
				with gradio.Blocks():
					lip_syncer_options.render()
				with gradio.Blocks():
					execution.render()
					execution_thread_count.render()
				with gradio.Blocks():
					download.render()
				with gradio.Blocks():
					state_manager.set_item('video_memory_strategy', 'tolerant')
					memory.render()
			with gradio.Column(scale = 11):
				with gradio.Blocks():
					benchmark.render()
	return layout


def listen() -> None:
	processors.listen()
	download.listen()
	face_enhancer_options.listen()
	lip_syncer_options.listen()
	execution.listen()
	execution_thread_count.listen()
	memory.listen()
	benchmark.listen()
	benchmark_options.listen()


def run(ui : gradio.Blocks) -> None:
	ui.launch(favicon_path = 'facefusion.ico', inbrowser = state_manager.get_item('open_browser'))
