from typing import List, get_args

from facefusion.processors.modules.ditto.types import DittoBackend, DittoRenderMode, DittoSourceMode

ditto_source_modes : List[DittoSourceMode] = list(get_args(DittoSourceMode))
ditto_backends : List[DittoBackend] = list(get_args(DittoBackend))
ditto_render_modes : List[DittoRenderMode] = list(get_args(DittoRenderMode))
