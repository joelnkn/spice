import logging
import os
from typing import Any
from llm_client import PromptManager
from utils import load_required_files

logger = logging.getLogger(__name__)

def run_analysis_step(llm_client: Any, prompt_dir: str, output_dir: str, run_name: str):
    """Run final analysis on the complete language
    Args:
        llm_client: LLM client instance for generation
        prompt_dir: Path to the prompts directory
        output_dir: Base output directory
        run_name: Name of the run
    """
    logger.info("Starting final language analysis")
    
    # Get memory_dir from final_iteration
    memory_dir = os.path.join(output_dir, run_name, 'languages', 'final_iteration', 'memory')
    
    if not os.path.exists(memory_dir):
        logger.error(f"Final iteration memory directory not found: {memory_dir}")
        return False
    
    files = load_required_files(memory_dir, {
        'phonology': 'phonology.txt',
        'grammar': 'grammar.txt',
        'lexicon': 'lexicon.csv',
    })
    if files is None:
        logger.error("Could not load required files for analysis")
        return False
    
    lex_section = f"""It has the following lexicon:\n\n=== START ===\n{files['lexicon']}\n=== END ==="""

    analysis_prompt_dir = os.path.join(prompt_dir, 'analysis')
    try:
        prompt = PromptManager.load_prompt(os.path.join(analysis_prompt_dir, 'feature_analysis.txt'))
    except Exception as e:
        logger.error(f"Could not load analysis prompt: {e}")
        return False
    
    kwargs = {
        'phonology': files['phonology'],
        'grammar': files['grammar'],
        'lexicon_section': lex_section,
    }
    
    logger.info("Generating final language analysis")
    _, analysis = llm_client.generate_and_extract(
        PromptManager.format_prompt(prompt, **kwargs),
        do_sleep=False
    )
    
    if analysis is None:
        logger.error("Failed to generate analysis")
        return False
        
    # Save analysis to output_dir/run_name/analysis
    analysis_dir = os.path.join(output_dir, run_name, 'analysis')
    os.makedirs(analysis_dir, exist_ok=True)
    
    with open(os.path.join(analysis_dir, 'features.txt'), 'w', encoding='utf-8') as f:
        f.write(analysis)
    
    logger.info(f"Final analysis completed and saved to {analysis_dir}")
    return True