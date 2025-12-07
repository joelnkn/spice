"""
Utility functions for synthetic language generation
"""
import os
import re

from synthetic.config import OUTPUT_DIR

def max_attempt_number(run_directory, prefix="attempt"):
    """Get the latest number in the given run directory matching the prefix pattern.
    
    Args:
        run_directory: Directory to search
        prefix: Prefix to match (default: "attempt")
    
    Returns:
        Maximum number found, or 0 if none found
    """
    max_attempt = 0
    if os.path.exists(run_directory):
        pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
        for name in os.listdir(run_directory):
            m = pattern.match(name)
            if m:
                num = int(m.group(1))
                if num > max_attempt:
                    max_attempt = num
    return max_attempt

def get_latest_target_id(target):
    dir = os.path.join(OUTPUT_DIR, target, 'languages')
    max_attempt = max_attempt_number(dir, "attempt")
    if max_attempt == 0:
        return None
    return f"attempt_{max_attempt}"

def get_latest_random_id(average_hamming_dist, num_in_group):
	dir = os.path.join(OUTPUT_DIR, 'random', f"{average_hamming_dist}_{num_in_group}", "languages")
	print(f"dir for latest random id: {dir}")
	max_attempt = max_attempt_number(dir, "attempt")
	if max_attempt == 0:
		return None
	print(f"latest random id: attempt_{max_attempt}")
	return f"attempt_{max_attempt}"

def get_new_target_id(target):
    dir = os.path.join(OUTPUT_DIR, target, "languages")
    max_attempt = max_attempt_number(dir, "attempt")
    return f"attempt_{max_attempt + 1}"

def get_new_random_id(average_hamming_dist, num_in_group):
	dir =  os.path.join(OUTPUT_DIR, 'random', f"{average_hamming_dist}_{num_in_group}", "languages")
	max_attempt = max_attempt_number(dir, "attempt")
	return f"attempt_{max_attempt + 1}"

def get_latest_target_iteration(target, lang_id):
    dir = os.path.join(OUTPUT_DIR, target, 'languages', lang_id, 'memory')
    return max_attempt_number(dir, "iter")

def get_latest_random_iteration(average_hamming_dist, num_in_group, lang_id):
	dir = os.path.join(OUTPUT_DIR, 'random', f"{average_hamming_dist}_{num_in_group}", 'languages', lang_id, 'memory')
	return max_attempt_number(dir, "iter")


