import csv
import importlib
import os
import json
import logging
import re
import shutil
from typing import Any, Dict, List, Optional

from llm_client import PromptManager

logger = logging.getLogger(__name__)


def create_llm_client(model='gemini-2.5-pro', max_tokens=8192, temperature=0.7, 
                     sleep_between_calls=30, debug=False, thinking_budget=1000, 
                     reasoning_effort='medium'):
    """Create an LLM client based on model name.
    
    Args:
        model: Model identifier (e.g., gemini-2.5-pro, gpt-4o, deepseek-ai/DeepSeek-R1)
        max_tokens: Maximum tokens for generation
        temperature: Temperature for sampling
        sleep_between_calls: Sleep time between API calls (seconds)
        debug: Enable debug mode with dummy responses
        thinking_budget: Thinking budget for models that support it (Gemini)
        reasoning_effort: Reasoning effort for OpenAI o-series models
        
    Returns:
        Initialized LLM client instance
    """
    from llm_client import LLMClientGemini, LLMClientDeepseek, LLMClientOpenAI, LLMClientMock
    
    if model.startswith('gemini'):
        return LLMClientGemini(
            model_checkpoint=model,
            max_tokens=max_tokens,
            temperature=temperature,
            sleep_between_calls=sleep_between_calls,
            debug=debug,
            thinking_budget=thinking_budget
        )
    elif model.startswith('deepseek'):
        return LLMClientDeepseek(
            model_checkpoint=model,
            max_tokens=max_tokens,
            temperature=temperature,
            sleep_between_calls=sleep_between_calls,
            debug=debug
        )
    elif model.startswith('o') or model.startswith('gpt-'):
        return LLMClientOpenAI(
            model_checkpoint=model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            temperature=temperature,
            sleep_between_calls=sleep_between_calls,
            debug=debug
        )
    elif model == 'mock-local':
        return LLMClientMock()
    else:
        raise ValueError(f"Unsupported model: {model}")


def clean_response(response: str, response_type: str = "text") -> str:
    """Clean LLM response by removing markdown formatting and extracting content."""
    if response_type == "csv":
        # Extract CSV content from markdown code blocks
        csv_pattern = r"```(?:csv)?\s*\n(.*?)\n```"
        matches = re.findall(csv_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        return response.strip()

    elif response_type == "json":
        # Extract JSON content from markdown code blocks
        json_pattern = r"```(?:json)?\s*\n(.*?)\n```"
        matches = re.findall(json_pattern, response, re.DOTALL)
        if matches:
            return matches[0].strip()
        return response.strip()

    return response.strip()


def alphabetize_csv_text(csv_text: str) -> str:
    """Alphabetize CSV entries while preserving the header."""
    lines = csv_text.strip().split("\n")
    if len(lines) <= 1:
        return csv_text

    header = lines[0]
    data_lines = [line for line in lines[1:] if line.strip()]

    # Sort data lines alphabetically
    data_lines.sort()

    return "\n".join([header] + data_lines)


def get_csv_text_n_entries(csv_text: str) -> int:
    """Count the number of data entries in CSV text (excluding header)."""
    lines = csv_text.strip().split("\n")
    # Count non-empty lines excluding the header
    return len([line for line in lines[1:] if line.strip()]) if len(lines) > 1 else 0


def load_required_files(memory_dir: str, required_files: dict) -> dict:
    """Load required files from memory directory."""
    files = {}

    for key, filename in required_files.items():
        # Try to find the file in the appropriate subdirectory
        file_path = os.path.join(memory_dir, key, filename)

        if not os.path.exists(file_path):
            logger.error(f"Required file not found: {file_path}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                files[key] = f.read()
            logger.info(f"Loaded {key}: {file_path}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None

    return files


def save_memory(content: str, memory_dir: str, filename: str, metadata: dict):
    """Save content and metadata to memory directory."""
    os.makedirs(memory_dir, exist_ok=True)

    # Save content
    content_path = os.path.join(memory_dir, filename)
    with open(content_path, "w", encoding="utf-8") as f:
        f.write(content)

    # Save metadata
    metadata_path = os.path.join(memory_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved {filename} and metadata to {memory_dir}")


# === Additional helper utilities for QA-enabled pipeline ===


def load_files_with_optional(
    memory_dir: str, required_files: dict, optional_files: dict
) -> dict:
    """Load required files plus optional files if they exist.

    Returns a dict containing all loaded file contents, or None if a required file is missing.
    """
    files = load_required_files(memory_dir, required_files)
    if files is None:
        return None
    for key, filename in optional_files.items():
        path = os.path.join(memory_dir, key, filename)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    files[key] = f.read()
                logger.info(f"Loaded optional {key}: {path}")
            except Exception as e:
                logger.warning(f"Failed loading optional {key} ({path}): {e}")
    return files


def save_memory_without_metadata(content: str, memory_dir: str, filename: str):
    """Save content file only (do not create/modify metadata.json)."""
    os.makedirs(memory_dir, exist_ok=True)
    path = os.path.join(memory_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info(f"Saved {filename} (no metadata) to {memory_dir}")


def save_individual_metadata(metadata: dict, memory_dir: str, filename: str):
    """Save a standalone metadata JSON (used for per-item outputs like individual translations)."""
    os.makedirs(memory_dir, exist_ok=True)
    path = os.path.join(memory_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved individual metadata {filename} to {memory_dir}")


def copy_folders(lang_dir: str, dst_dir: str, folder_names: list) -> None:
    """Copy specified folders from lang_dir/memory to destination directory, and copy metadata.json from lang_dir."""
    memory_dir = os.path.join(lang_dir, "memory")
    if not os.path.exists(memory_dir):
        raise FileNotFoundError(f"Memory directory not found: {memory_dir}")

    os.makedirs(dst_dir, exist_ok=True)

    for folder in folder_names:
        src = os.path.join(memory_dir, folder)
        dst = os.path.join(dst_dir, folder)

        if os.path.exists(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            logger.info(f"Copied folder '{folder}' from {memory_dir} to {dst_dir}")
        else:
            logger.warning(f"Folder '{folder}' not found in memory directory: {src}")

    # Also copy metadata.json if it exists in lang_dir
    src_metadata = os.path.join(lang_dir, "metadata.json")
    dst_metadata = os.path.join(dst_dir, "metadata.json")
    if os.path.exists(src_metadata):
        shutil.copy2(src_metadata, dst_metadata)
        logger.info(f"Copied metadata.json from {lang_dir} to {dst_dir}")

def extract_required_lexicon(args, llm_client: Any) -> Optional[str]:
    """
    Uses LLM to select the lexicon entries needed for the translation sentences.
    
    Saves result to:
        memory_dir/iter_<iteration>/required_lexicon.csv

    Args:
        args: must contain:
            - args.prompt_dir
            - args.memory_dir
            - args.iteration
            - args.sentences (newline-separated English sentences)
        llm_client: your LLM wrapper with generate_and_extract() or similar.

    Returns:
        Path to required_lexicon.csv, or None on failure.
    """

    # ---------- Paths ----------
    lexicon_path = os.path.join(args.memory_dir, "lexicon", "lexicon.csv")
    prompt_path = os.path.join(args.prompt_dir, "lexicon", "lex_extraction.txt")
    output_dir = os.path.join(args.memory_dir, f"iter_{args.iteration}")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "required_lexicon.csv")

    # ---------- Load lexicon.csv ----------
    if not os.path.exists(lexicon_path):
        logger.error(f"Lexicon file not found at {lexicon_path}")
        return None

    with open(lexicon_path, "r", encoding="utf-8") as f:
        lexicon_text = f.read()

    # ---------- Load extraction prompt ----------
    try:
        base_prompt = PromptManager.load_prompt(prompt_path)
    except Exception as e:
        logger.error(f"Failed to load lexicon extraction prompt: {e}")
        return None

    # ---------- Format prompt ----------
    prompt = PromptManager.format_prompt(
        base_prompt,
        lexicon_csv=lexicon_text,
        sentences=args.translation_sentence
    )

    logger.debug(f"Lexicon extraction prompt:\n{prompt}")

    # ---------- Call LLM ----------
    logger.info(f"Using LLM to extract required lexicon entries for iteration {args.iteration}")
    _, llm_output = llm_client.generate_and_extract(
        prompt,
        do_sleep=False
    )

    if llm_output is None:
        logger.error("LLM returned null for required lexicon extraction")
        return None

    # ---------- Write output CSV ----------
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(llm_output.strip())
        logger.info(f"Saved required lexicon to {output_path}")
    except Exception as e:
        logger.error(f"Error writing required lexicon CSV: {e}")
        return None

    return output_path

def check_new_word_conflicts(new_words: List[Dict[str, str]], args) -> List[Dict[str, str]]:
    """
    Compare newly generated words against the existing lexicon and detect 
    collisions where the same word form already exists but with different meaning or POS.

    Args:
        new_words: list of dicts with keys: "word", "pos", "translation"
        args: namespace with memory_dir pointing to the language directory

    Returns:
        List of conflict objects:
        [
            {
                "word": "<string>",
                "existing_pos": "<pos>",
                "existing_translation": "<translation>",
                "new_pos": "<pos>",
                "new_translation": "<translation>",
                "issue": "pos_mismatch" | "translation_mismatch" | "both_mismatch"
            }
        ]
    """
    lexicon_path = os.path.join(args.memory_dir, "lexicon", "lexicon.csv")

    if not os.path.exists(lexicon_path):
        # No lexicon → no conflicts possible
        return []

    # Load existing lexicon
    existing = {}  # word → list of (pos, translation)
    with open(lexicon_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            word = row["form"].strip()
            pos = row["pos"].strip()
            translation = row["translation"].strip()
            if word not in existing:
                existing[word] = []
            existing[word].append((pos, translation))

    conflicts = []

    for nw in new_words:
        w = nw["word"].strip()
        new_pos = nw["pos"].strip()
        new_trans = nw["translation"].strip()

        if w not in existing:
            continue  # no conflict

        for (exist_pos, exist_trans) in existing[w]:
            pos_mismatch = exist_pos != new_pos
            trans_mismatch = exist_trans != new_trans and new_trans not in exist_trans and exist_trans not in new_trans

            if pos_mismatch or trans_mismatch:
                conflicts.append({
                    "word": w,
                    "existing_pos": exist_pos,
                    "existing_translation": exist_trans,
                    "new_pos": new_pos,
                    "new_translation": new_trans,
                    "issue": (
                        "both_mismatch" if (pos_mismatch and trans_mismatch)
                        else "pos_mismatch" if pos_mismatch
                        else "translation_mismatch"
                    )
                })

    return conflicts

def get_specification_dir(prompt_dir: str, lang_name: str, random: bool) -> str:
    """
    Get the directory path for specifications based on language type.
    prompt_dir is expected to be in conlanger/prompts
    base_specifications is in conlanger/base_specifications
    """
    base_path = os.path.normpath(os.path.join(prompt_dir, '..', 'base_specifications'))

    if random:
        # Expect lang_name like 'low_0', 'high_2', etc.
        m = re.match(r"(low|medium|high)_(\d+)$", lang_name)
        if not m:
            raise ValueError(f"Invalid random lang_name format: {lang_name}")
        group = m.group(1)
        path = os.path.join(base_path, "random", f"{group}")
    else:
        # Actual language
        path = os.path.join(base_path, "target")

    return path

def get_specification_value(prompt_dir: str, lang_name: str, random: bool, filename: str) -> str:
    base_path = get_specification_dir(prompt_dir, lang_name, random)

    if random:
        group, idx = re.match(r"(low|high)_(\d+)$", lang_name).groups()
        idx = int(idx)
        path = os.path.join(base_path, filename)
        with open(path, "r", encoding="utf-8") as f:
            specs = json.load(f)
        if not isinstance(specs, list):
            raise ValueError(f"Expected a list in {path}, got {type(specs)}")
        if idx >= len(specs):
            raise IndexError(f"Index {idx} out of range for {group} features (len={len(specs)})")
        val = specs[idx]
    else:
        # Actual language
        path = os.path.join(base_path, filename)
        with open(path, "r", encoding="utf-8") as f:
            specs = json.load(f)
        val = specs.get(lang_name, None)
        if val is None:
            raise ValueError(f"Language '{lang_name}' not found in {os.path.basename(specs)}")
    return val

def load_feature_vector(prompt_dir: str, lang_name: str, random: bool) -> dict:
    """
    Load feature vector for this run.
    """
    return get_specification_value(prompt_dir, lang_name, random, "feature_vectors.json")

def load_affix(prompt_dir: str, lang_name: str, random: bool) -> dict:
    """
    Load affix specifications for this run.
    """
    return get_specification_value(prompt_dir, lang_name, random, "affixes.json")

def load_lexicon(args) -> str:
    """
    Load lexicon CSV for this run and write it to memory_dir/lexicon/lexicon.csv
    - For target: loads from base_specifications/target/lexicons.csv
    - For random: loads from base_specifications/random/[group]/[group]_[idx].csv
    
    Returns the CSV content as a string.
    """
    base_path = get_specification_dir(args.prompt_dir, args.lang_name, args.random)
    
    if args.random:
        # Expect lang_name like 'low_0', 'medium_2', etc.
        group, idx = re.match(r"(low|high)_(\d+)$", args.lang_name).groups()
        path = os.path.join(base_path, f"{group}_{idx}.csv")
        with open(path, "r", encoding="utf-8") as f:
            csv_content = f.read()
    else:
        # Actual language - load CSV file from target/lexicons.csv
        path = os.path.join(base_path, f"{args.lang_name}.csv")
        with open(path, "r", encoding="utf-8") as f:
            csv_content = f.read()
    
    # Write to memory_dir/lexicon/lexicon.csv
    lex_dir = os.path.join(args.memory_dir, "lexicon")
    os.makedirs(lex_dir, exist_ok=True)
    lex_path = os.path.join(lex_dir, "lexicon.csv")
    with open(lex_path, "w", encoding="utf-8") as f:
        f.write(csv_content)
    
    logger.info(f"Loaded and saved lexicon to {lex_path}")
    return csv_content

def load_orthography_rules(prompt_dir: str, lang_name: str, random: bool) -> str:
    """
    Load orthography rules for a language and combine with baseline.
    
    Args:
        prompt_dir: Path to prompts directory
        lang_name: Language name (e.g., "arabic" or "low_0")
        random: Whether this is a random language
    
    Returns:
        Combined orthography text (baseline + language-specific rules)
    """
    orthography_path = os.path.normpath(os.path.join(prompt_dir, '..', 'base_specifications', 'orthography.py'))
    spec = importlib.util.spec_from_file_location("orthography", orthography_path)
    ortho_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ortho_mod)
    
    BASELINE = getattr(ortho_mod, "BASELINE", "")
    RULES_PER_LANGUAGE = getattr(ortho_mod, "RULES_PER_LANGUAGE", {})
    
    # Use 'random' as key if random, otherwise use lang_name
    ortho_key = 'random' if random else lang_name
    
    logger.info(f"Loading orthography rules for key: {ortho_key}")
    
    rules = RULES_PER_LANGUAGE.get(ortho_key, None)
    if rules is None:
        raise ValueError(
            f"'{ortho_key}' not found in RULES_PER_LANGUAGE. "
            "Please add it to orthography.py (including a 'random' profile if needed)."
        )
        
    logger.info(f"Loaded orthography rules for {ortho_key}")
    
    ortho_text = BASELINE.strip() + "\n• " + rules.strip()
    return ortho_text


def append_lexicon_entries_from_input(required_lex_csv: str, full_lexicon_csv: str, input_sentences: str) -> str:
    """
    Append lexicon entries from full lexicon to required lexicon for words in input sentences.
    
    Parses input sentences (separated by newlines), extracts all words, and appends matching
    rows from the full lexicon to the required lexicon without creating duplicates.
    
    Args:
        required_lex_csv: CSV string with already extracted entries (from LLM)
        full_lexicon_csv: Full lexicon CSV with format "form,pos,translation"
        input_sentences: Sentences separated by newlines
        
    Returns:
        CSV string with required entries + new entries from full lexicon (no duplicates)
    """
    if not input_sentences.strip() or not full_lexicon_csv.strip():
        return required_lex_csv
    
    # Parse required lexicon CSV to get existing forms
    required_lines = required_lex_csv.strip().split('\n') if required_lex_csv.strip() else []
    header = required_lines[0] if required_lines else "form,pos,translation"
    
    # Build set of forms already in required lexicon
    existing_forms = set()
    required_rows = [header]
    
    for line in required_lines[1:]:
        if not line.strip():
            continue
        parts = line.split(',')
        if len(parts) >= 1:
            form = parts[0].strip()
            existing_forms.add(form.lower())  # Store lowercase for case-insensitive comparison
            required_rows.append(line)
    
    # Parse full lexicon CSV
    full_lines = full_lexicon_csv.strip().split('\n')
    if not full_lines:
        return required_lex_csv
    
    # Build mapping of forms to rows in full lexicon
    full_lexicon_rows = {}  # lowercase form -> row mapping
    
    for line in full_lines[1:]:  # Skip header
        if not line.strip():
            continue
        parts = line.split(',')
        if len(parts) >= 1:
            form = parts[0].strip()
            full_lexicon_rows[form.lower()] = line
    
    # Extract words from input sentences
    words_in_input = set()
    for sentence in input_sentences.split('\n'):
        if not sentence.strip():
            continue
        # Split by whitespace and common punctuation
        sentence_words = re.findall(r'\b\w+\b', sentence.lower())
        words_in_input.update(sentence_words)
    
    logger.info(f"Found {len(words_in_input)} unique words in input sentences")
    logger.info(f"Found {len(existing_forms)} entries already in required lexicon")
    logger.info(f"Found {len(full_lexicon_rows)} entries in full lexicon")
    
    # Append matching entries from full lexicon (if not already in required)
    added_count = 0
    
    for word in sorted(words_in_input):
        word_lower = word.lower()
        # Check if word is in full lexicon and NOT already in required lexicon
        if word_lower in full_lexicon_rows and word_lower not in existing_forms:
            required_rows.append(full_lexicon_rows[word_lower])
            existing_forms.add(word_lower)  # Track to prevent duplicates
            added_count += 1
    
    logger.info(f"Appended {added_count} new entries from full lexicon")
    logger.info(f"Final required lexicon has {len(required_rows) - 1} entries")
    
    return '\n'.join(required_rows)
