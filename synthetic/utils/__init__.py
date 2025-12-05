"""
Utility functions for synthetic language generation
"""
import os
import re

from synthetic.config import OUTPUT_DIR

def max_attempt_number(run_directory):
    """Get the latest language ID in the given run directory."""
    max_attempt = 0
    if os.path.exists(run_directory):
        for name in os.listdir(run_directory):
            m = re.match(r"attempt_(\d+)$", name)
            if m:
                num = int(m.group(1))
                if num > max_attempt:
                    max_attempt = num
    return max_attempt

def get_latest_language_id(run_directory):
    max_attempt = max_attempt_number(run_directory)
    if max_attempt == 0:
        return None
    return f"attempt_{max_attempt}"

def get_new_language_id(run_directory):
    max_attempt = max_attempt_number(run_directory)
    return f"attempt_{max_attempt + 1}"

def get_latest_target_language_id(target):
    return get_latest_language_id(os.path.join(OUTPUT_DIR, target, 'languages'))

def get_latest_random_language_id(average_hamming_dist, num_in_group):
    return get_latest_language_id(os.path.join(OUTPUT_DIR, 'random', f"f{average_hamming_dist}_{num_in_group}", 'languages'))

def get_new_target_language_id(target):
    return get_new_language_id(os.path.join(OUTPUT_DIR, target, 'languages'))
                    
def get_new_random_language_id(average_hamming_dist, num_in_group):
    return get_new_language_id(os.path.join(OUTPUT_DIR, 'random', f"f{average_hamming_dist}_{num_in_group}", 'languages')) 

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
