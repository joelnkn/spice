#!/usr/bin/env python3
"""
ConlangCrafter: AI-Powered Constructed Language Generator

This script generates constructed languages (conlangs) using AI models.
It supports grammar, lexicon generation, and translation.
"""

import os
import sys
import time
import logging
import uuid
import json
from argparse import ArgumentParser
from dotenv import load_dotenv

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from synthetic.typology.orthography import ORTHOGRAPHY_BASELINE

# Load environment variables from .env file
load_dotenv()

from pipeline_steps import run_grammar_step, run_lexicon_step, run_translation_step
from utils import copy_folders, create_llm_client
from cleanup import append_sentences_to_valid_translations, extract_new_vocabulary, extract_new_grammar_rules, append_new_words_to_lexicon, add_new_rules_to_grammar

logger = logging.getLogger(__name__)


def generate_language_id():
    """Generate a unique language ID."""
    return str(uuid.uuid4())[:8]


def setup_directories(output_dir, language_id):
    """Set up directories for a specific language."""
    lang_dir = os.path.join(output_dir, 'languages', language_id)
    memory_dir = os.path.join(lang_dir, 'memory')
    logs_dir = os.path.join(lang_dir, 'logs')
    
    os.makedirs(lang_dir, exist_ok=True)
    os.makedirs(memory_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    return lang_dir, memory_dir, logs_dir


def setup_logging(output_file: str):
    """Set up logging configuration."""
    logdir = os.path.dirname(output_file)
    os.makedirs(logdir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_file),
            logging.StreamHandler()
        ]
    )


def save_metadata(lang_dir, language_id, args):
    """Save metadata about the language generation."""
    metadata = {
        'language_id': language_id,
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model': args.model,
        'steps': args.steps.split(','),
        'custom_constraints': args.custom_constraints,
        'parameters': {
            'temperature': getattr(args, 'temperature', None),
            'max_tokens': getattr(args, 'max_tokens', None),
        }
    }
    
    metadata_file = os.path.join(lang_dir, 'metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    return metadata_file


def get_args():
    """Parse command line arguments."""
    parser = ArgumentParser(description='Generate constructed languages using AI')
    
    # Model settings
    parser.add_argument(
        '--model',
        default='gemini-2.5-pro',
        help=(
            'Model identifier to use. Examples: gemini-2.5-pro, gemini-1.5-flash, '
            'o4-mini, gpt-4o, gpt-5, deepseek-ai/DeepSeek-R1. '
            'Any valid provider model string is accepted.'
        ),
    )
    parser.add_argument('--max-tokens', type=int, default=8192,
                       help='Maximum tokens for generation')
    parser.add_argument('--temperature', type=float, default=0.7,
                       help='Temperature for sampling')
    parser.add_argument('--thinking-budget', type=int, default=1000,
                       help='Thinking budget for models that support it')
    parser.add_argument('--reasoning-effort', default='medium', choices=['low','medium','high'],
                       help='Reasoning effort for OpenAI o-series')
    parser.add_argument('--sleep-between-calls', type=float, default=30,
                       help='Sleep time between API calls (seconds)')

    # QA settings
    parser.add_argument('--qa-enabled', action='store_true',
                        help='Enable QA self-refine (critic/amend) loop for supported steps')
    parser.add_argument('--self-refine-steps', type=int, default=3,
                        help='Number of QA self-refine (critic/amend) cycles')
    parser.add_argument('--qa-threshold', type=float, default=None,
                        help='Global passing score threshold (1–10 scale) overriding all per-step thresholds if set')
    parser.add_argument('--qa-threshold-grammar', type=float, default=8.0,
                        help='Passing score threshold (1–10 scale) for grammar QA')
    parser.add_argument('--qa-threshold-lexicon', type=float, default=8.0,
                        help='Passing score threshold (1–10 scale) for lexicon QA')
    parser.add_argument('--qa-threshold-translation', type=float, default=8.0,
                        help='Passing score threshold (1–10 scale) for translation QA')
    parser.add_argument('--continue-qa', action='store_true',
                        help='If a QA report exists, continue from previous iterations and append results')
    
    # Pipeline settings
    parser.add_argument('--steps', default='grammar,lexicon,translation',
                       help='Comma-separated list of steps to run')
    parser.add_argument('--custom-constraints', 
                       help='Custom constraints for language generation')
    parser.add_argument('--translation-sentence', 
                       default='The quick brown fox jumps over the lazy dog.',
                       help='Sentence to translate into the constructed language')
    
    # Generation parameters
    parser.add_argument('--gram-n-questions', type=int, default=10,
                       help='Number of grammar questions')
    parser.add_argument('--gram-n-answers', type=int, default=5,
                       help='Number of grammar answer options')
    parser.add_argument('--gram-scale-size', type=int, default=5,
                       help='Grammar scale size')
    
    parser.add_argument('--lexicon-min-entries', type=int, default=50,
                       help='Minimum lexicon entries')
    parser.add_argument('--lexicon-n-per-iter', type=int, default=15,
                       help='Lexicon entries per iteration')
    parser.add_argument('--lexicon-max-iters', type=int, default=5,
                       help='Maximum lexicon iterations')
    parser.add_argument('--lexicon-extra-sleep', type=float, default=30,
                       help='Extra sleep for lexicon generation')
    
    # Paths
    parser.add_argument('--prompt-dir', default='prompts',
                       help='Directory containing prompt templates')
    parser.add_argument('--output-dir', default='output',
                       help='Output directory for generated languages')
    parser.add_argument('--iteration', action='store_true',
                       help='Whether to use iterations to produce a language across multiple Conglanger calls')
    parser.add_argument('--lang-id', default=None,
                        help='ID for already created language during stabilization')
    
    # Debug mode
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode with dummy responses')
    
    return parser.parse_args()


def main():
    """Main function to run the ConlangCrafter pipeline."""
    args = get_args()

    language_id = generate_language_id() if args.lang_id is None else args.lang_id
    print(f"Generating language with ID: {language_id}")

    # Set up directories
    lang_dir, memory_dir, logs_dir = setup_directories(args.output_dir, language_id)
    args.memory_dir = memory_dir
    
    # Ensure orthography/orthography.txt exists inside memory_dir
    ortho_dir = os.path.join(memory_dir, "orthography")
    os.makedirs(ortho_dir, exist_ok=True)

    ortho_path = os.path.join(ortho_dir, "orthography.txt")
    if not os.path.exists(ortho_path):
        with open(ortho_path, "w", encoding="utf-8") as f:
            f.write(ORTHOGRAPHY_BASELINE.strip() + "\n")
        logger.info(f"Wrote orthography baseline to {ortho_path}")
    else:
        logger.info(f"Found existing orthography at {ortho_path}")

    log_file = os.path.join(logs_dir, 'pipeline.log')
    setup_logging(log_file)
    
    logger.info(f"Starting language generation with ID: {language_id}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Steps: {args.steps}")
    
    # Save metadata
    metadata_file = save_metadata(lang_dir, language_id, args)
    logger.info(f"Metadata saved to: {metadata_file}")
    
    # Initialize LLM client
    llm_client = create_llm_client(
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        sleep_between_calls=args.sleep_between_calls,
        debug=args.debug,
        thinking_budget=args.thinking_budget,
        reasoning_effort=args.reasoning_effort
    )
    
    # Parse and run steps
    steps = [step.strip() for step in args.steps.split(',')]
    step_functions = {
        'grammar': run_grammar_step,
        'lexicon': run_lexicon_step,
        'translation': run_translation_step,
    }
    
    print(f"\nRunning steps: {', '.join(steps)}")
    
    for i, step in enumerate(steps):
        if step not in step_functions:
            logger.error(f"Unknown step: {step}")
            continue
        
        print(f"\n=== Running {step} step ({i+1}/{len(steps)}) ===")
        logger.info(f"Starting {step} step")
        
        # Add translation sentence to args
        if step == 'translation':
            args.translation_input_sentence = args.translation_sentence
        
        try:
            result = step_functions[step](args, llm_client)
            if not result:
                logger.error(f"Step {step} failed")
                break
            logger.info(f"Completed {step} step")
        except Exception as e:
            logger.error(f"Error in {step} step: {e}")
            break
        
        # Sleep between steps
        if i < len(steps) - 1 and not args.debug:
            logger.info(f"Sleeping for 30 seconds...")
            time.sleep(30)
    
    print(f"\nLanguage generation completed!")
    print(f"Results saved in: {lang_dir}")
    logger.info(f"Language generation completed for ID: {language_id}")
    
    # Post-processing based on iteration mode
    if args.iteration:
        # Iteration mode: creating evolving language snapshots
        if 'translation' not in steps:
            # Case 1: Initial language generation (no translation step)
            # Create iter_0 with grammar and lexicon
            iteration = 0
            logger.info("Initial language generation - creating iter_0")
            new_iter_dir = os.path.join(lang_dir, f"iter_{iteration}")
            os.makedirs(new_iter_dir, exist_ok=False)
            copy_folders(lang_dir, new_iter_dir, ['grammar', 'lexicon'])
            logger.info(f"Saved initial snapshot: {new_iter_dir}")
        else:
            # Case 2: Translation step with iteration (adapting language)
            # First, apply new words and grammar rules from translation
            logger.info("Processing translation outputs for iteration...")
            
            # Extract and append new words to lexicon
            new_words = extract_new_vocabulary(lang_dir)
            if new_words:
                append_new_words_to_lexicon(new_words, args)
                logger.info(f"Added {len(new_words)} new words to lexicon")
            
            # Integrate new grammar rules using LLM
            new_rules = extract_new_grammar_rules(lang_dir)
            if new_rules:
                logger.info(f"Integrating {len(new_rules)} new grammar rules...")
                result = add_new_rules_to_grammar(new_rules, args, llm_client)
                if result:
                    logger.info(f"Successfully integrated grammar rules")
                else:
                    logger.warning("Failed to integrate grammar rules")
            
            # Now create the next iteration snapshot
            iter_dirs = [
                name for name in os.listdir(lang_dir)
                if name.startswith("iter_") and name[len("iter_"):].isdigit()
            ]
            
            if not iter_dirs:
                logger.error("No iter_0 found! Translation step requires initial language generation first.")
                raise RuntimeError("Cannot run translation in iteration mode without iter_0")
            
            iteration = max(int(name[len("iter_"):]) for name in iter_dirs) + 1
            logger.info(f"Adapting language - creating iter_{iteration}")
            
            new_iter_dir = os.path.join(lang_dir, f"iter_{iteration}")
            os.makedirs(new_iter_dir, exist_ok=False)
            copy_folders(lang_dir, new_iter_dir, ['grammar', 'lexicon', 'translation'])
            logger.info(f"Saved iteration snapshot: {new_iter_dir}")
    
    elif 'translation' in steps:
        # Non-iteration mode: stabilized language, just append new words
        logger.info("Processing translation outputs (stabilized language mode)...")
        
        new_words = extract_new_vocabulary(lang_dir)
        if new_words:
            append_new_words_to_lexicon(new_words, args)
            logger.info(f"Added {len(new_words)} new words to lexicon")
        
        # Check for grammar rules and warn if any exist
        new_rules = extract_new_grammar_rules(lang_dir)
        if new_rules:
            logger.warning(f"Found {len(new_rules)} new grammar rules but skipping integration (stabilized language mode)")
        else:
            logger.info("No new grammar rules found (as expected for stabilized language)")
            
        append_sentences_to_valid_translations(args.memory_dir)
    
if __name__ == '__main__':
    main()