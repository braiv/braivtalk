from facefusion.types import Locales

LOCALES : Locales =\
{
	'en':
	{
		'help':
		{
			'model': 'choose the model responsible for syncing the lips',
			'pure_motion': 'refine with live-portrait',
			'motion_damping': 'reduce the stitcher correction strength to calm jittery live-portrait motion',
			'motion_smoothing': 'smooth the live-portrait driver motion across frames',
			'motion_mask_mode': 'choose the mask strategy for live-portrait paste-back (off, box, hybrid)',
			'mask_blur': 'control the feather softness of the pure-motion mask edge',
			'mask_erode': 'shrink the pure-motion mask inward to avoid ghost chins',
			'mask_expand': 'expand the pure-motion mask outward in all directions',
			'chin_expand': 'expand the lower mask downward to cover chin mismatches',
			'occlusion_dilate': 'relax the occlusion mask outward for pure-motion blending',
			'occlusion_blur': 'soften the occlusion mask edge for smoother lower-face blending',
			'expressiveness': 'scale the lip transfer strength for more or less mouth movement',
			'weight': 'specify the degree of weight applied to the lips'
		},
		'uis':
		{
			'model_dropdown': 'LIP SYNCER MODEL',
			'pure_motion_slider': 'LIP SYNCER PURE MOTION',
			'motion_damping_slider': 'LIP SYNCER MOTION DAMPING',
			'motion_smoothing_checkbox': 'LIP SYNCER MOTION SMOOTHING',
			'motion_mask_mode_dropdown': 'LIP SYNCER MASK MODE',
			'mask_blur_slider': 'LIP SYNCER MASK BLUR',
			'mask_erode_slider': 'LIP SYNCER MASK ERODE',
			'mask_expand_slider': 'LIP SYNCER MASK EXPAND',
			'chin_expand_slider': 'LIP SYNCER CHIN EXPAND',
			'occlusion_dilate_slider': 'LIP SYNCER OCCLUSION DILATE',
			'occlusion_blur_slider': 'LIP SYNCER OCCLUSION BLUR',
			'expressiveness_slider': 'LIP SYNCER EXPRESSIVENESS',
			'weight_slider': 'LIP SYNCER WEIGHT'
		}
	}
}
