from typing import List, Sequence, get_args

from facefusion.common_helper import create_float_range, create_int_range
from facefusion.processors.modules.lip_syncer.types import LipSyncerModel, LipSyncerMotionMaskMode

lip_syncer_models : List[LipSyncerModel] = list(get_args(LipSyncerModel))
lip_syncer_motion_mask_modes : List[LipSyncerMotionMaskMode] = list(get_args(LipSyncerMotionMaskMode))

lip_syncer_pure_motion_range : Sequence[float] = create_float_range(0.0, 1.5, 0.25)
lip_syncer_motion_damping_range : Sequence[float] = create_float_range(0.0, 1.0, 0.05)
lip_syncer_crop_stabilization_range : Sequence[float] = create_float_range(0.0, 1.0, 0.05)
lip_syncer_weight_range : Sequence[float] = create_float_range(0.0, 1.0, 0.05)
lip_syncer_mask_blur_range : Sequence[float] = create_float_range(0.0, 1.0, 0.05)
lip_syncer_mask_erode_range : Sequence[int] = create_int_range(0, 40, 1)
lip_syncer_mask_expand_range : Sequence[int] = create_int_range(0, 40, 1)
lip_syncer_chin_expand_range : Sequence[int] = create_int_range(0, 40, 1)
lip_syncer_occlusion_dilate_range : Sequence[int] = create_int_range(0, 30, 1)
lip_syncer_occlusion_blur_range : Sequence[float] = create_float_range(0.0, 1.0, 0.05)
lip_syncer_expressiveness_range : Sequence[float] = create_float_range(0.5, 2.0, 0.05)
