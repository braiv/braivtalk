from typing import List, get_args

from facefusion.processors.modules.ditto.types import DittoBackend, DittoCompositeGeometryMode, DittoCropPrepMode, DittoRenderMode, DittoSourceMode

ditto_source_modes : List[DittoSourceMode] = list(get_args(DittoSourceMode))
ditto_backends : List[DittoBackend] = list(get_args(DittoBackend))
ditto_render_modes : List[DittoRenderMode] = list(get_args(DittoRenderMode))
ditto_crop_prep_modes : List[DittoCropPrepMode] = list(get_args(DittoCropPrepMode))
ditto_composite_geometry_modes : List[DittoCompositeGeometryMode] = list(get_args(DittoCompositeGeometryMode))
