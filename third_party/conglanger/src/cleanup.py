import csv
import json
import logging
import os
from typing import Optional, List, Dict, Any
from llm_client import PromptManager
from utils import load_required_files

logger = logging.getLogger(__name__)

def append_new_words_to_lexicon(new_words: List[Dict[str, str]], args) -> str:
    """Append new words to the lexicon CSV file in memory directory.
    
    Args:
        new_words: List of {word: translation} dicts
        args: Namespace with memory_dir
        
    Returns:
        Path to the updated lexicon CSV file
    """
    lexicon_csv_path = os.path.join(args.memory_dir, 'lexicon', 'lexicon.csv')
    
    if not new_words or not os.path.exists(lexicon_csv_path):
        return lexicon_csv_path
    
    # Append to the lexicon file in memory_dir
    with open(lexicon_csv_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for word_dict in new_words:
            # Expecting {'word': ..., 'pos': ..., 'translation': ...}
            word = word_dict.get('word')
            pos = word_dict.get('pos', '')
            translation = word_dict.get('translation', '')
            writer.writerow([word, pos, translation, '', '', ''])
    
    logger.info(f"Appended {len(new_words)} new words to {lexicon_csv_path}")
    
    return lexicon_csv_path


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
    
def update_metadata_qa(lang_dir):
    """Update metadata with QA score from translation_qa.json.
    
    Reads the overall_score from memory/translation/translation_qa.json and
    updates the running_qa sum and count in metadata.json.
    
    Args:
        lang_dir: Path to the language directory (contains memory/)
    """
    memory_dir = os.path.join(lang_dir, "memory")
    translation_qa_path = os.path.join(memory_dir, 'translation', 'translation_qa.json')
    
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
    
    # Extract overall_score from final_qa
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
    
    # Update running QA sum and count
    metadata["running_qa"] = metadata.get("running_qa", 0) + overall_score
    metadata["running_qa_count"] = metadata.get("running_qa_count", 0) + 1
    metadata["average_qa"] = metadata["running_qa"] / metadata["running_qa_count"]
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    logger.info(f"Updated metadata.json with running_qa={metadata['running_qa']}, count={metadata['running_qa_count']}, average={metadata['average_qa']:.2f}")

def extract_new_vocabulary(lang_dir) -> List[Dict[str, str]]:
    """Extract new words from all sentences in translation.json in lang_dir/memory/translation/translation.json.
    Args:
        lang_dir: Path to the language directory (contains memory/)
    Returns:
        List of {word: translation} dicts from all sentences
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
    new_words = []
    for sentence in sentences:
        for word_dict in sentence.get('new_words', []):
            new_words.append(word_dict)
    logger.info(f"Extracted {len(new_words)} new words from all translations")
    update_metadata_value(lang_dir, "num_new_words", len(new_words))
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

def append_sentences_to_valid_translations(memory_dir) -> str:
    """Append all sentences from translation.json to valid_translations.json in memory_dir/translation.
    Creates valid_translations.json if it does not exist.
    Args:
        memory_dir: Path to the memory directory
    Returns:
        Path to the updated valid_translations.json file
    """
    translation_dir = os.path.join(memory_dir, 'translation')
    translation_json_path = os.path.join(translation_dir, 'translation.json')
    valid_translations_path = os.path.join(translation_dir, 'valid_translations.json')

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
    logger.info(f"Appended {len(sentences)} sentences to {valid_translations_path}")
    return valid_translations_path