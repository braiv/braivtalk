LOCALES =\
{
	'help':
	{
		'source_mode': 'select ditto source mode: video_native feeds each frame, image_anchor uses a single reference',
		'backend': 'select ditto inference backend',
		'render_mode': 'select ditto render/compositing mode',
		'registration_crop_mode': 'choose how sampled registration frames are normalized before Ditto inference',
		'live_crop_mode': 'choose how live per-frame crops are normalized before Ditto inference',
		'composite_geometry_mode': 'choose how Ditto render output is mapped back into the final composite'
	},
	'uis':
	{
		'source_mode_dropdown': 'DITTO SOURCE MODE',
		'backend_dropdown': 'DITTO BACKEND',
		'render_mode_dropdown': 'DITTO RENDER MODE',
		'registration_crop_mode_dropdown': 'DITTO REGISTRATION CROP MODE',
		'live_crop_mode_dropdown': 'DITTO LIVE CROP MODE',
		'composite_geometry_mode_dropdown': 'DITTO COMPOSITE GEOMETRY'
	}
}


def get(key_path : str) -> str:
	parts = key_path.split('.')
	node = LOCALES
	for part in parts:
		if isinstance(node, dict) and part in node:
			node = node[part]
		else:
			return key_path
	return node if isinstance(node, str) else key_path
