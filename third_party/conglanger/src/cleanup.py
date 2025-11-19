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
            for word, translation in word_dict.items():
                writer.writerow([word, translation, '', '', '', ''])
    
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
    # Load existing grammar using load_required_files
    files = load_required_files(args.memory_dir, {'grammar': 'grammar.txt'})
    if files is None:
        logger.error("Could not load grammar file")
        return None
    
    existing_grammar = files['grammar']
    
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
    formatted_prompt = PromptManager.format_prompt(prompt, grammar=existing_grammar, new_rules=formatted_rules)
    
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

def extract_new_vocabulary(args) -> List[Dict[str, str]]:
    """Extract new words from the most recent sentence in translation.json.
    
    Args:
        args: Namespace with memory_dir
    
    Returns:
        List of {word: translation} dicts from the last sentence only
    """
    translation_json_path = os.path.join(args.memory_dir, 'translation', 'translation.json')
    
    if not os.path.exists(translation_json_path):
        logger.warning(f"Translation file not found: {translation_json_path}")
        return []
    
    with open(translation_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentences = data.get('sentences', [])
    if not sentences:
        logger.warning("No sentences found in translation.json")
        return []
    
    # Only extract from the LAST sentence (most recently added)
    last_sentence = sentences[-1]
    new_words = last_sentence.get('new_words', [])
    
    logger.info(f"Extracted {len(new_words)} new words from latest translation")
    return new_words


def extract_new_grammar_rules(args) -> List[str]:
    """Extract new grammar rules from the most recent sentence in translation.json.
    
    Args:
        args: Namespace with memory_dir
    
    Returns:
        List of rule strings from the last sentence only
    """
    translation_json_path = os.path.join(args.memory_dir, 'translation', 'translation.json')
    
    if not os.path.exists(translation_json_path):
        logger.warning(f"Translation file not found: {translation_json_path}")
        return []
    
    with open(translation_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sentences = data.get('sentences', [])
    if not sentences:
        logger.warning("No sentences found in translation.json")
        return []
    
    # Only extract from the LAST sentence (most recently added)
    last_sentence = sentences[-1]
    new_rules = []
    for rule_obj in last_sentence.get('new_grammar_rules', []):
        if 'rule' in rule_obj:
            new_rules.append(rule_obj['rule'])
    
    logger.info(f"Extracted {len(new_rules)} new grammar rules from latest translation")
    return new_rules