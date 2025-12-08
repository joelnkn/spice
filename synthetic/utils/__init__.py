"""
Utility functions for synthetic language generation
"""
import os
import re
import json

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

def get_target_memory_dir(target, lang_id):
	dir = os.path.join(OUTPUT_DIR, target, 'languages', lang_id, 'memory')
	return dir

def get_random_memory_dir(average_hamming_dist, num_in_group, lang_id):
	dir = os.path.join(OUTPUT_DIR, 'random', f"{average_hamming_dist}_{num_in_group}", 'languages', lang_id, 'memory')
	return dir

def clean_translations(dir):
	"""
	Clean valid_translations.json by:
	1. Converting conlang_sentence to lowercase
	2. Removing POS tags (adj, noun, verb, particle, case_marker, etc.)
	3. Ensuring proper spacing
	4. Copying all other fields except new_words
	5. Saving to cleaned_translations.json in same directory
	
	Args:
		dir: Path to directory containing valid_translations.json
	
	Returns:
		Path to cleaned_translations.json or None if failed
	"""
	valid_translations_path = os.path.join(dir, "valid_translations.json")
	cleaned_translations_path = os.path.join(dir, "cleaned_translations.json")
	
	if not os.path.exists(valid_translations_path):
		print(f"valid_translations.json not found at {valid_translations_path}")
		return None
	
	# Comprehensive POS tags to remove
	pos_tags = {
		# Major POS
		"adj", "adjective", "adjectival",
		"noun", "n", "nom", "nominal",
		"verb", "v", "vb",
		"adv", "adverb", "adverbial",
		"prep", "preposition", "adposition",
		"pron", "pronoun",
		"det", "determiner",
		"conj", "conjunction", "conjunctive",
		"interj", "interjection",
		"num", "numeral", "number", "numeral",
		"art", "article",
		"particle", "participle",
		"case_marker", "marker", "affix", "suffix", "prefix", "infix",
		"tense", "aspect", "mood",
		"gender", "number_", "person",
		"comparative", "superlative",
		"past", "present", "future",
		"singular", "plural", "dual",
		"positive", "negative",
		"imperative", "subjunctive", "conditional",
		"gerund", "infinitive",
		"clause", "phrase"
	}
	
	try:
		with open(valid_translations_path, "r", encoding="utf-8") as f:
			data = json.load(f)
		
		sentences = data.get("sentences", [])
		cleaned_sentences = []
		
		for sentence_obj in sentences:
			# Copy all fields except new_words
			cleaned_obj = {
				k: v for k, v in sentence_obj.items() 
				if k != "new_words"
			}
			
			# Clean the conlang_sentence
			conlang_sentence = cleaned_obj.get("conlang_sentence", "")
			cleaned = conlang_sentence.lower()
			
			# Remove POS tags - match whole words only
			for pos_tag in pos_tags:
				pattern = rf"\b{re.escape(pos_tag)}\b"
				cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
			
			# Clean up spacing: remove extra spaces, normalize to single spaces
			cleaned = re.sub(r"\s+", " ", cleaned).strip()
			
			# Update conlang_sentence with cleaned version
			cleaned_obj["conlang_sentence"] = cleaned
			
			if cleaned:  # Only add non-empty sentences
				cleaned_sentences.append(cleaned_obj)
		
		# Save cleaned data
		with open(cleaned_translations_path, "w", encoding="utf-8") as f:
			json.dump({"sentences": cleaned_sentences}, f, ensure_ascii=False, indent=2)
		
		print(f"Cleaned {len(cleaned_sentences)} sentences and saved to {cleaned_translations_path}")
		return cleaned_translations_path
		
	except json.JSONDecodeError as e:
		print(f"Error parsing JSON from {valid_translations_path}: {e}")
		return None
	except Exception as e:
		print(f"Error reading/processing {valid_translations_path}: {e}")
		return None

