import logging
import os
from llm_client import PromptManager
from utils import load_required_files

logger = logging.getLogger(__name__)

def extract_new_vocabulary_and_rules(translation_json_path: str) -> dict:
    """Extract new words and grammar rules from translation.json.
    
    Returns:
        dict with keys:
            'new_words': list of {word: translation} dicts
            'new_grammar_rules': list of rule strings
    """
    import json
    
    with open(translation_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    new_words = []
    new_rules = []
    
    for sentence in data.get('sentences', []):
        # Extract new words
        for word_dict in sentence.get('new_words', []):
            new_words.append(word_dict)
        
        # Extract grammar rules (just the rule text, not justification)
        for rule_obj in sentence.get('new_grammar_rules', []):
            if 'rule' in rule_obj:
                new_rules.append(rule_obj['rule'])
    
    return {
        'new_words': new_words,
        'new_grammar_rules': new_rules
    }

def run_analysis_step(args, llm_client):
    """Run final analysis on the complete language."""
    logger.info("Starting final language analysis")
    
    files = load_required_files(args.memory_dir, {
        'phonology': 'phonology.txt',
        'grammar': 'grammar.txt',
        'lexicon': 'lexicon.csv',
    })
    if files is None:
        logger.error("Could not load required files for analysis")
        return
        
    prompt_dir = os.path.join(args.prompt_dir, 'analysis')
    try:
        prompt = PromptManager.load_prompt(os.path.join(prompt_dir, 'feature_analysis.txt'))
    except Exception as e:
        logger.error(f"Could not load analysis prompt: {e}")
        return False
    
    # Extract new words and grammar rules from translation
    translation_path = os.path.join(args.memory_dir, 'translation', 'translation.json')
    new_content = {'new_words': [], 'new_grammar_rules': []}
    if os.path.exists(translation_path):
        try:
            new_content = extract_new_vocabulary_and_rules(translation_path)
            logger.info(f"Extracted {len(new_content['new_words'])} new words and {len(new_content['new_grammar_rules'])} new grammar rules from translation")
        except Exception as e:
            logger.warning(f"Could not extract new content from translation: {e}")
    
    # Build lexicon section with new words if available
    lex_base = files['lexicon']
    if len(new_content['new_words']) > 0:
        new_words_text = "\n\nAdditional words from translation:\n"
        for word_dict in new_content['new_words']:
            for word, translation in word_dict.items():
                new_words_text += f"{word},{translation}\n"
        lex_section = f"""It has the following lexicon:\n\n=== START ===\n{lex_base}{new_words_text}\n=== END ==="""
    else:
        lex_section = f"""It has the following lexicon:\n\n=== START ===\n{lex_base}\n=== END ==="""
    
    # Build grammar with new rules if available
    grammar_base = files['grammar']
    if len(new_content['new_grammar_rules']) > 0:
        new_rules_text = "\n\n=== NEW GRAMMAR RULES FROM TRANSLATION ===\n"
        for i, rule in enumerate(new_content['new_grammar_rules'], 1):
            new_rules_text += f"{i}. {rule}\n"
        grammar_with_new_rules = grammar_base + new_rules_text
    else:
        grammar_with_new_rules = grammar_base
    
    kwargs = {
        'phonology': files['phonology'],
        'grammar': grammar_with_new_rules,
        'lexicon_section': lex_section,
    }
    
    logger.info("Generating final language analysis")
    _, analysis = llm_client.generate_and_extract(
        PromptManager.format_prompt(prompt, **kwargs),
        do_sleep=False
    )
    
    if analysis is None:
        logger.error("Failed to generate analysis")
        return
        
    # Save analysis
    analysis_dir = os.path.join(args.memory_dir, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    with open(os.path.join(analysis_dir, 'features.txt'), 'w', encoding='utf-8') as f:
        f.write(analysis)
    
    logger.info("Final analysis completed and saved")