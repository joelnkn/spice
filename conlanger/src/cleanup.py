import json
import logging
import os
import re
from typing import Optional, List, Dict, Any

import pandas as pd
from llm_client import PromptManager
from utils import load_required_files

logger = logging.getLogger(__name__)

def update_lexicon_with_new_words(new_words: List[Dict[str, str]], args) -> str:
    """
    Update the main lexicon.csv with new words and store new words for this iteration.

    Expected new_words format:
    [
        {"word": "<form>", "pos": "<pos>", "translation": "<english>"},
        ...
    ]

    Saves:
        memory_dir/iter_<iteration>/new_words.json
    Updates:
        memory_dir/lexicon/lexicon.csv

    Returns:
        Path to updated lexicon.csv
    """

    # ----------------------------------------------------
    # Paths
    # ----------------------------------------------------
    lexicon_path = os.path.join(args.memory_dir, "lexicon", "lexicon.csv")
    iter_dir = os.path.join(args.memory_dir, f"iter_{args.iteration}")
    os.makedirs(iter_dir, exist_ok=True)
    new_words_path = os.path.join(iter_dir, "new_words.json")

    # ----------------------------------------------------
    # Save new words (JSON) at iteration level
    # ----------------------------------------------------
    try:
        with open(new_words_path, "w", encoding="utf-8") as f:
            json.dump(new_words, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved new words for iteration {args.iteration} to {new_words_path}")
    except Exception as e:
        logger.error(f"Failed to save new_words.json: {e}")

    # If no new words, no need to modify lexicon
    if not new_words:
        logger.info("No new words generated; lexicon unchanged.")
        return lexicon_path

    # ----------------------------------------------------
    # Load existing lexicon
    # ----------------------------------------------------
    if not os.path.exists(lexicon_path):
        logger.error(f"Lexicon not found at {lexicon_path}")
        return lexicon_path

    df = pd.read_csv(lexicon_path, dtype=str).fillna("")

    # Get set of existing forms for quick lookup
    existing_forms = set(df['form'].str.lower())

    # ----------------------------------------------------
    # Normalize and collect new entries (skip if already in lexicon, update if compatible)
    # ----------------------------------------------------
    new_entries = []
    updates = []
    skipped_count = 0
    for entry in new_words:
        form = entry.get("word", "").strip().lower()
        pos = entry.get("pos", "").strip().lower()
        translation = entry.get("translation", "").strip().lower()

        if not form or form in ("suffix", "prefix", "infix", "affix"):
            continue

        # Check if word already exists in lexicon
        if form in existing_forms:
            # Find the existing row
            existing_row = df[df['form'].str.lower() == form].iloc[0]
            old_pos = existing_row['pos'].lower()
            old_translation = existing_row['translation'].lower()
            
            # If POS matches and old translation is in new translation, update it
            if pos == old_pos and old_translation in translation:
                updates.append((form, pos, translation))
                logger.debug(f"Updating '{form}' translation from '{old_translation}' to '{translation}'")
            else:
                skipped_count += 1
                logger.debug(f"Skipping '{form}' - already in lexicon with different POS or incompatible translation")
            continue

        new_entries.append({"form": form, "pos": pos, "translation": translation})

    df_new = pd.DataFrame(new_entries)

    # If no new entries and no updates, log and return
    if df_new.empty and not updates:
        logger.info(f"No new words to add (skipped {skipped_count} words already in lexicon)")
        return lexicon_path

    # ----------------------------------------------------
    # Apply updates to existing entries
    # ----------------------------------------------------
    for form, pos, translation in updates:
        df.loc[df['form'].str.lower() == form, 'translation'] = translation
        df.loc[df['form'].str.lower() == form, 'pos'] = pos

    # ----------------------------------------------------
    # Merge + dedupe by "form"
    # ----------------------------------------------------
    df_full = pd.concat([df, df_new], ignore_index=True)
    df_full = df_full.drop_duplicates(subset=["form"], keep="first")

    # ----------------------------------------------------
    # Save updated lexicon
    # ----------------------------------------------------
    df_full.to_csv(lexicon_path, index=False)

    logger.info(
        f"Added {len(df_new)} new words, updated {len(updates)} existing words "
        f"({len(df_full) - len(df)} net unique additions) "
        f"to lexicon at {lexicon_path}"
    )

    return lexicon_path


def add_new_rules_to_grammar(new_rules: List[str], args, llm_client: Any) -> Optional[str]:
    """
    Use LLM to intelligently integrate new grammar rules into the existing grammar in memory directory.
    
    Args:
        new_rules: List of rule strings
        args: Namespace with prompt_dir and memory_dir
        llm_client: LLM client instance for generation
    
    Returns:
        Path to the updated grammar.txt file, or None if failed
    """
    # Load existing grammar and orthography using load_required_files
    files = load_required_files(args.memory_dir, {'grammar': 'grammar.txt', 'orthography': 'orthography.txt'})
    if files is None:
        logger.error("Could not load grammar or orthography file")
        return None

    existing_grammar = files['grammar']
    orthography = files['orthography']

    # Format new rules for the prompt
    formatted_rules = "\n".join([f"- {rule}" for rule in new_rules])

    # Load and format the prompt using PromptManager static method
    grammar_prompt_dir = os.path.join(args.prompt_dir, 'grammar')
    try:
        prompt = PromptManager.load_prompt(os.path.join(grammar_prompt_dir, 'apply_new_rules.txt'))
    except Exception as e:
        logger.error(f"Could not load apply_new_rules prompt: {e}")
        return None

    # Format the prompt with context
    formatted_prompt = PromptManager.format_prompt(
        prompt,
        grammar=existing_grammar,
        new_rules=formatted_rules,
        orthography=orthography
    )
    
    # Log the prompt being sent
    logger.debug(f"Prompt for grammar rule integration:\n{formatted_prompt}")
    
    # Generate updated grammar using LLM
    logger.info(f"Using LLM to integrate {len(new_rules)} new rules into grammar")
    _, updated_grammar = llm_client.generate_and_extract(
        formatted_prompt,
        do_sleep=False
    )
    
    if updated_grammar is None:
        logger.error("Failed to generate updated grammar")
        return None
    
    # Log the generated output
    logger.debug(f"Generated updated grammar:\n{updated_grammar}")
    
    # Write updated grammar back to memory directory
    grammar_path = os.path.join(args.memory_dir, "grammar", "grammar.txt")
    
    with open(grammar_path, 'w', encoding='utf-8') as f:
        f.write(updated_grammar)
    
    logger.info(f"Updated grammar with {len(new_rules)} new rules at {grammar_path}")
    return grammar_path

def update_metadata_value(lang_dir, key: str, value):
    """Helper to update a value (str or int) in metadata.json in lang_dir."""
    metadata_path = os.path.join(lang_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata.json: {e}")
    metadata[key] = value
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Updated metadata.json with {key}={value}")
    
def update_metadata_with_translation_qa(lang_dir, iteration):
    """Update metadata with QA score from translation_qa.json.
    
    Reads the overall_score from memory/iter_{iteration}/translation/translation_qa.json and
    updates the running_qa sum and count in metadata.json.
    
    Args:
        lang_dir: Path to the language directory (contains memory/)
    """
    memory_dir = os.path.join(lang_dir, "memory")
    translation_qa_path = os.path.join(memory_dir, f"iter_{iteration}", "translation", 'translation_qa.json')
    
    if not os.path.exists(translation_qa_path):
        logger.warning(f"Translation QA file not found: {translation_qa_path}")
        return
    
    # Read QA score from translation_qa.json
    try:
        with open(translation_qa_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
    except Exception as e:
        logger.warning(f"Could not load translation_qa.json: {e}")
        return
    
    # Extract info from final_qa
    final_qa = qa_data.get('final_qa') or {}
    if not isinstance(final_qa, dict):
        logger.warning(f"final_qa is not a dict in translation_qa.json: {type(final_qa)}")
        return
    
    overall_score = final_qa.get('overall_score', 0)
    if overall_score == 0:
        logger.warning(f"No overall_score found in translation_qa.json")
        return
    
    # Update metadata
    metadata_path = os.path.join(lang_dir, "metadata.json")
    metadata = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load metadata.json: {e}")
            
    if final_qa.get('conflicts') or final_qa.get('overall_score') < 5:
        metadata["num_invalid_batches"] = metadata.get("num_invalid_batches", 0) + 1
        metadata["invalid_batches"].append(iteration)
        return True
    
    # Update running QA sum and count
    metadata["running_qa"] = metadata.get("running_qa", 0) + overall_score
    metadata["running_qa_count"] = metadata.get("running_qa_count", 0) + 1
    metadata["average_qa"] = metadata["running_qa"] / metadata["running_qa_count"]
    metadata["num_valid_batches"] = metadata.get("num_valid_batches", 0) + 1
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Updated metadata.json with running_qa={metadata['running_qa']}, count={metadata['running_qa_count']}, average={metadata['average_qa']:.2f}")
    
    return False

def extract_new_vocabulary(lang_dir, iteration) -> List[Dict[str, str]]:
    """Extract new words from all sentences in translation.json in lang_dir/memory/translation/translation.json.
    Args:
        lang_dir: Path to the language directory (contains memory/)
    Returns:
        List of {word: translation} dicts from all sentences
    """
    memory_dir = os.path.join(lang_dir, "memory")
    translation_json_path = os.path.join(memory_dir, f"iter_{iteration}", 'translation', 'translation.json')
    if not os.path.exists(translation_json_path):
        logger.warning(f"Translation file not found: {translation_json_path}")
        return []
    with open(translation_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sentences = data.get('sentences', [])
    if not sentences:
        logger.warning("No sentences found in translation.json")
        return []
    new_words = []
    for sentence in sentences:
        for word_dict in sentence.get('new_words', []):
            new_words.append(word_dict)
    logger.info(f"Extracted {len(new_words)} new words from all translations")
    return new_words

def extract_new_grammar_rules(lang_dir) -> List[str]:
    """Extract new grammar rules from all sentences in translation.json in lang_dir/memory/translation/translation.json.
    Args:
        lang_dir: Path to the language directory (contains memory/)
    Returns:
        List of rule strings from all sentences
    """
    memory_dir = os.path.join(lang_dir, "memory")
    translation_json_path = os.path.join(memory_dir, 'translation', 'translation.json')
    if not os.path.exists(translation_json_path):
        logger.warning(f"Translation file not found: {translation_json_path}")
        return []
    with open(translation_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sentences = data.get('sentences', [])
    if not sentences:
        logger.warning("No sentences found in translation.json")
        return []
    new_rules = []
    for sentence in sentences:
        for rule_obj in sentence.get('new_grammar_rules', []):
            if 'rule' in rule_obj:
                new_rules.append(rule_obj['rule'])
    logger.info(f"Extracted {len(new_rules)} new grammar rules from all translations")
    update_metadata_value(lang_dir, "num_new_grammar_rules", len(new_rules))
    return new_rules

def append_sentences_to_valid_translations(memory_dir, iteration, input_sentences, batch_size=20) -> str:
    """Append all sentences from translation.json to valid_translations.json in memory_dir/translation.
    Creates valid_translations.json if it does not exist.
    Each sentence group includes the iteration number that translated those sentences.
    Args:
        memory_dir: Path to the memory directory
        iteration: The iteration number that produced these translations
    Returns:
        Path to the updated valid_translations.json file
    """
    english_sentences = input_sentences.split("\n")
    translation_dir = os.path.join(memory_dir, f"iter_{iteration}", 'translation')
    translation_json_path = os.path.join(translation_dir, 'translation.json')
    valid_translations_path = os.path.join(memory_dir, 'valid_translations.json')

    # Load sentences from translation.json
    if not os.path.exists(translation_json_path):
        logger.warning(f"Translation file not found: {translation_json_path}")
        return valid_translations_path
    with open(translation_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    sentences = data.get('sentences', [])
    if not sentences:
        logger.warning("No sentences found in translation.json")
        return valid_translations_path

    # Add iteration number to each sentence
    for i, sentence in enumerate(sentences):
        sentence['iteration'] = iteration
        sentence['index_in_iteration'] = i
        sentence['global_index'] = iteration * batch_size + i
        sentence['english_sentence'] = english_sentences[i]

    # Load or create valid_translations.json
    if os.path.exists(valid_translations_path):
        with open(valid_translations_path, 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
        valid_sentences = valid_data.get('sentences', [])
    else:
        valid_sentences = []

    # Append all sentences from translation.json
    valid_sentences.extend(sentences)
    with open(valid_translations_path, 'w', encoding='utf-8') as f:
        json.dump({'sentences': valid_sentences}, f, ensure_ascii=False, indent=2)
    logger.info(f"Appended {len(sentences)} sentences from iteration {iteration} to {valid_translations_path}")
    return valid_translations_path