from facefusion.types import Locales

LOCALES : Locales =\
{
	'en':
	{
		'help':
		{
			'model': 'choose the model responsible for syncing the lips',
			'pure_motion': 'refine with live-portrait',
			'motion_smoothing': 'smooth the live-portrait driver motion across frames',
			'weight': 'specify the degree of weight applied to the lips'
		},
		'uis':
		{
			'model_dropdown': 'LIP SYNCER MODEL',
			'pure_motion_slider': 'LIP SYNCER PURE MOTION',
			'motion_smoothing_checkbox': 'LIP SYNCER MOTION SMOOTHING',
			'weight_slider': 'LIP SYNCER WEIGHT'
		}
	}
}
