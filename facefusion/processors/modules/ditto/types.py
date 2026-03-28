from typing import Any, List, Literal, Optional, TypedDict

from numpy.typing import NDArray

from facefusion.types import AudioFrame, Mask, VisionFrame

DittoInputs = TypedDict('DittoInputs',
{
	'reference_vision_frame' : VisionFrame,
	'source_vision_frames' : Optional[List[VisionFrame]],
	'source_audio_frame' : Optional[AudioFrame],
	'source_voice_frame' : Optional[AudioFrame],
	'target_vision_frame' : VisionFrame,
	'temp_vision_frame' : VisionFrame,
	'temp_vision_mask' : Mask,
	'frame_number' : int,
	'debug_mask' : bool
}, total = False)

DittoSourceMode = Literal['video_native', 'image_anchor']
DittoBackend = Literal['onnx', 'trt', 'pytorch']
DittoRenderMode = Literal['native_putback', 'facefusion_composite']
DittoCropPrepMode = Literal['ditto_native', 'facefusion_crop']
DittoCompositeGeometryMode = Literal['ditto_transform', 'facefusion_affine', 'warp_to_facefusion']
