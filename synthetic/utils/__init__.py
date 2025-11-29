"""
Utility functions for synthetic language generation
"""
def load_metadata(lang_dir):
	"""Load metadata.json from a language directory, or return {} if missing."""
	import os
	import json
	metadata_path = os.path.join(lang_dir, "metadata.json")
	if os.path.exists(metadata_path):
		try:
			with open(metadata_path, "r", encoding="utf-8") as f:
				return json.load(f)
		except Exception as e:
			print(f"Warning: could not load metadata.json: {e}")
	return {}
