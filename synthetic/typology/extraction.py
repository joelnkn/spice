"""Feature extraction for synthetic languages using WALS-style analysis."""

import logging
import os
from typing import Any

from synthetic.typology.features import parse_analysis_text_to_feature_dict, save_feature_files
from synthetic.conglanger import PromptManager, load_required_files
from synthetic.config import OUTPUT_DIR, PROMPT_DIR

logger = logging.getLogger(__name__)


def extract_features(llm_client: Any, run_name: str, language_id: str, 
                    prompt_dir: str = None, output_dir: str = None):
    """Extract linguistic features from a generated language using WALS-style analysis.
    
    Args:
        llm_client: LLM client instance for generation
        run_name: Name of the run
        language_id: ID of the language to analyze
        prompt_dir: Path to the prompts directory (defaults to third_party/conglanger/prompts)
        output_dir: Base output directory (defaults to synthetic/data/processed)
        
    Returns:
        bool: True if extraction succeeded, False otherwise
    """
    # Use defaults from config if not provided
    if prompt_dir is None:
        prompt_dir = PROMPT_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    logger.info(f"Starting feature extraction for language {language_id}")
    
    # Get memory_dir from run_name/languages/language_id
    memory_dir = os.path.join(output_dir, run_name, 'languages', language_id, 'memory')
    
    if not os.path.exists(memory_dir):
        logger.error(f"Memory directory not found: {memory_dir}")
        return False
    
    files = load_required_files(memory_dir, {
        'grammar': 'grammar.txt',
        'lexicon': 'lexicon.csv',
        'orthography': 'orthography.txt',
    })
    if files is None:
        logger.error("Could not load required files for feature extraction")
        return False
    
    lex_section = f"""It has the following lexicon:\n\n=== START ===\n{files['lexicon']}\n=== END ==="""

    typology_prompt_dir = os.path.join(prompt_dir, 'typology')
    try:
        prompt = PromptManager.load_prompt(os.path.join(typology_prompt_dir, 'feature_analysis.txt'))
    except Exception as e:
        logger.error(f"Could not load analysis prompt: {e}")
        return False
    
    kwargs = {
        'orthography': files['orthography'],
        'grammar': files['grammar'],
        'lexicon_section': lex_section,
    }
    
    logger.info("Generating linguistic feature analysis")
    _, analysis = llm_client.generate_and_extract(
        PromptManager.format_prompt(prompt, **kwargs),
        do_sleep=False
    )
    
    if analysis is None:
        logger.error("Failed to generate feature analysis")
        return False
        
    # Save analysis to output_dir/run_name/languages/language_id/analysis
    analysis_dir = os.path.join(output_dir, run_name, 'languages', language_id, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    features_file = os.path.join(analysis_dir, 'features.txt')
    with open(features_file, 'w', encoding='utf-8') as f:
        f.write(analysis)
    
    logger.info(f"Feature extraction completed and saved to {features_file}")
    
    logger.info("Parsing features and generating JSON...")
    feature_dict = parse_analysis_text_to_feature_dict(analysis)
    language_dir = os.path.join(output_dir, run_name, 'languages', language_id)
    saved_files = save_feature_files(language_dir, feature_dict)
    
    if saved_files:
        logger.info(f"Feature JSON saved: {saved_files}")
    else:
        logger.warning("Failed to save feature JSON")
    
    return True
