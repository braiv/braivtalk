LOCALES =\
{
	'help':
	{
		'source_mode': 'select ditto source mode: video_native feeds each frame, image_anchor uses a single reference',
		'backend': 'select ditto inference backend',
		'render_mode': 'select ditto render/compositing mode'
	},
	'uis':
	{
		'source_mode_dropdown': 'DITTO SOURCE MODE',
		'backend_dropdown': 'DITTO BACKEND',
		'render_mode_dropdown': 'DITTO RENDER MODE'
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
