#!/usr/bin/env python3
"""
ConlangCrafter: AI-Powered Constructed Language Generator

This script generates constructed languages (conlangs) using AI models.
It supports grammar, lexicon generation, and translation.
"""

import os
import time
import logging
import uuid
import json
from argparse import ArgumentParser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from pipeline_steps import run_affix_step, run_lexicon_step, run_translation_step
from utils import create_llm_client
from cleanup import append_sentences_to_valid_translations, extract_new_vocabulary, extract_new_grammar_rules, add_new_rules_to_grammar, update_lexicon_with_new_words, update_metadata_value, update_metadata_with_translation_qa

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
    """Save metadata about the language generation.
    
    If metadata.json already exists, just update the 'continued_at' timestamp.
    Otherwise, create a new metadata file with full information.
    """
    metadata_file = os.path.join(lang_dir, 'metadata.json')
    
    if os.path.exists(metadata_file):
        # Load existing metadata and add continued_at timestamp
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            metadata['continued_last_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            if args.iteration in metadata['invalid_batches']:
                metadata['invalid_batches'].remove(args.iteration)
                metadata['num_invalid_batches'] -= 1
            metadata['started_at_iters'].append(args.iteration)
            
        except Exception as e:
            logger.warning(f"Could not load existing metadata, creating new: {e}")
            metadata = {
                'language_id': language_id,
                'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
                'model': args.model,
                'steps': args.steps.split(','),
                'custom_constraints': args.custom_constraints,
                'parameters': {
                    'temperature': getattr(args, 'temperature', None),
                    'max_tokens': getattr(args, 'max_tokens', None),
                },
                'started_at_iters': [],
                'num_invalid_batches': 0,
                'invalid_batches': []
            }
    else:
        # Create new metadata file
        metadata = {
            'language_id': language_id,
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model': args.model,
            'steps': args.steps.split(','),
            'custom_constraints': args.custom_constraints,
            'parameters': {
                'temperature': getattr(args, 'temperature', None),
                'max_tokens': getattr(args, 'max_tokens', None),
            },
            'started_at_iters': [],
            'num_invalid_batches': 0,
            'invalid_batches': []
        }
    
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
    parser.add_argument('--qa-threshold-affix', type=float, default=8.0,
                        help='Passing score threshold (1–10 scale) for affix QA')
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
    
    # New arguments
    parser.add_argument(
        '--iteration',
        type=int,
        default=-1,
        help=(
            'Translation iteration index. '
            '-1: no translation (e.g., initial lexicon generation only). '
            '>= 0: run translation for this iteration index.'
        )
    )
    parser.add_argument('--lang-id', default=None,
                        help='ID for already created language during stabilization')
    parser.add_argument('--random', action='store_true',
                        help='Indicates if the language is part of a random language group')
    parser.add_argument('--lang-name', default=None,
                        help='Name of the language being generated')
    
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
    args.lang_dir = lang_dir
    args.memory_dir = memory_dir

    # Logging
    log_file = os.path.join(logs_dir, 'pipeline.log')
    setup_logging(log_file)
    
    logger.info(f"Starting language generation with ID: {language_id}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Steps: {args.steps}")
    logger.info(f"Iteration: {args.iteration}")
    
    # Save metadata
    if args.iteration == 0 and 'translation' in args.steps:
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
        'affix': run_affix_step,
        'lexicon': run_lexicon_step,
        'translation': run_translation_step,
    }
    
    print(f"\nRunning steps: {', '.join(steps)}")
    
    if args.iteration == -1:
        # No translation expected in this mode
        if 'translation' in steps:
            logger.error("iteration = -1 but 'translation' step requested. Remove 'translation' from --steps or set iteration >= 0.")
            raise RuntimeError("Invalid configuration: translation step with iteration = -1")
    else:
        # Translation mode (iteration >= 0): we expect translation step
        if 'translation' not in steps:
            logger.error("iteration >= 0 but 'translation' step not present in --steps.")
            raise RuntimeError("Invalid configuration: iteration >= 0 requires 'translation' in --steps")

    
    for i, step in enumerate(steps):
        if step not in step_functions:
            logger.error(f"Unknown step: {step}")
            continue
        
        print(f"\n=== Running {step} step ({i+1}/{len(steps)}) ===")
        logger.info(f"Starting {step} step")
        
        # Add translation sentence to args
        if step == 'translation':
            args.input_sentences = args.translation_sentence
        
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
    
    # 1) No translation → nothing to do here except maybe lexicon QA
    if 'translation' not in steps:
        logger.info("No translation step in this run; skipping translation post-processing.")
        return

    # 2) Translation run (iteration >= 0) → handle new vocabulary + conflicts
    logger.info("Processing translation outputs...")

    # Extract new words from translation outputs (assuming QA/amend already done)
    # IMPORTANT: we now assume extract_new_vocabulary works from memory_dir
    new_words = extract_new_vocabulary(lang_dir, args.iteration)

    if not new_words:
        logger.info("No new words extracted from translation.")
        # Even if no new words, we can still append sentences that passed QA
        append_sentences_to_valid_translations(args.memory_dir, args.iteration, args.input_sentences)
        update_metadata_with_translation_qa(lang_dir, args.iteration)
        return
    
    # Still update QA metadata, but do not accept this batch into the stable dataset
    has_conflicts = update_metadata_with_translation_qa(lang_dir, args.iteration)
    if has_conflicts:
        logger.info("Conflicts detected; new words will not be added to lexicon.")
        return
    
    # If we reach here: no conflicts → safe to update lexicon + valid translations
    update_metadata_value(lang_dir, "num_new_words", len(new_words))
    update_lexicon_with_new_words(new_words, args)
    logger.info(f"Added {len(new_words)} new words to lexicon.")

    append_sentences_to_valid_translations(args.memory_dir, args.iteration, args.input_sentences)
    logger.info("Appended sentences to valid translations.")
    
if __name__ == '__main__':
    main()