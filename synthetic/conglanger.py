"""Wrapper module for conglanger to enable clean imports."""

import os
import sys
import importlib
from synthetic.generation.custom_constraints import BASE 
from synthetic.config import CONGLANGER_PATH, OUTPUT_DIR, PROMPT_DIR
from synthetic.utils.logger import setup_logger

# Add conglanger to path
_conglanger_src = os.path.join(os.path.dirname(__file__), '../third_party/conglanger/src')
if _conglanger_src not in sys.path:
    sys.path.insert(0, _conglanger_src)

# Import modules dynamically
_llm_client = importlib.import_module('llm_client')
_utils = importlib.import_module('utils')
_pipeline_steps = importlib.import_module('pipeline_steps')

# Re-export classes and functions
PromptManager = _llm_client.PromptManager
LLMClientGemini = _llm_client.LLMClientGemini
LLMClientDeepseek = _llm_client.LLMClientDeepseek
LLMClientOpenAI = _llm_client.LLMClientOpenAI

clean_response = _utils.clean_response
alphabetize_csv_text = _utils.alphabetize_csv_text
get_csv_text_n_entries = _utils.get_csv_text_n_entries
copy_folders = _utils.copy_folders
load_required_files = _utils.load_required_files
create_llm_client = _utils.create_llm_client

run_grammar_step = _pipeline_steps.run_grammar_step
run_lexicon_step = _pipeline_steps.run_lexicon_step
run_translation_step = _pipeline_steps.run_translation_step


def run_conglanger(
    run_name=None,
    model="gemini-2.5-flash-lite", # prob should change back to gemini-2.5-pro
    steps=("grammar", "lexicon", "translation"),
    custom_constraints=BASE,
    translation_sentence="Hello, world!",
    max_tokens=8192,
    temperature=0.7,
    thinking_budget=1000,
    reasoning_effort="low",
    sleep_between_calls=0,
    qa_enabled=True,
    self_refine_steps=2,
    qa_threshold=7.0,
    qa_thresholds_per_step=None,
    iteration=False,
    prompt_dir=None,
    output_dir=None,
    lang_id=None,
    debug=False,
    extra_args=None,
):
    """Wrapper for running Conglanger's run_pipeline.py as a subprocess.
    
    This calls the CLI tool as a subprocess, which is useful for:
    - Running in a separate process
    - Using the CLI's full argument parsing
    - Isolating execution environment
    """ 
    # Use defaults from config if not provided
    if prompt_dir is None:
        prompt_dir = PROMPT_DIR
    if output_dir is None:
        output_dir = OUTPUT_DIR
    
    # Logger setup
    logger = setup_logger("generate_language")

    # Resolve paths
    if not os.path.exists(CONGLANGER_PATH):
        raise FileNotFoundError(
            f"Conglanger entry point not found at: {CONGLANGER_PATH}"
        )

    os.makedirs(output_dir, exist_ok=True)

    run_name = run_name or "unnamed_run"
    output_dir = os.path.join(output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Build base command
    cmd = [
        "python",
        CONGLANGER_PATH,
        "--model",
        model,
        "--steps",
        ",".join(steps),
        "--output-dir",
        output_dir,
        "--max-tokens",
        str(max_tokens),
        "--temperature",
        str(temperature),
        "--sleep-between-calls",
        str(sleep_between_calls),
        "--prompt-dir",
        prompt_dir,
    ]

    if custom_constraints:
        cmd += ["--custom-constraints", custom_constraints]
    if translation_sentence:
        cmd += ["--translation-sentence", translation_sentence]
    if reasoning_effort:
        cmd += ["--reasoning-effort", reasoning_effort]
    if thinking_budget:
        cmd += ["--thinking-budget", str(thinking_budget)]
    if iteration:
        cmd += ["--iteration"]
    if lang_id:
        cmd += ["--lang-id", str(lang_id)]

    # QA loop configuration
    if qa_enabled:
        cmd.append("--qa-enabled")
        cmd += ["--self-refine-steps", str(self_refine_steps)]

        if qa_threshold is not None:
            cmd += ["--qa-threshold", str(qa_threshold)]

        if qa_thresholds_per_step:
            for step, threshold in qa_thresholds_per_step.items():
                cmd += [f"--qa-threshold-{step}", str(threshold)]

    # Debug mode
    if debug:
        cmd.append("--debug")

    # Add extra args
    if extra_args:
        cmd += extra_args

    # Run subprocess
    import subprocess
    
    logger.info(f"Running Conglanger pipeline for run: {run_name}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Output directory: {output_dir}")
    
    # Run with real-time output streaming
    result = subprocess.run(cmd, check=True)
    
    if result.returncode != 0:
        logger.error(f"Conglanger pipeline failed with exit code {result.returncode}")
        raise RuntimeError(f"Conglanger pipeline failed")

    logger.info("Conglanger pipeline completed successfully.")

    return output_dir


__all__ = [
    # LLM clients
    'PromptManager',
    'LLMClientGemini',
    'LLMClientDeepseek',
    'LLMClientOpenAI',
    'create_llm_client',
    
    # Utilities
    'clean_response',
    'alphabetize_csv_text',
    'get_csv_text_n_entries',
    'copy_folders',
    'load_required_files',
    
    # Pipeline steps

    'run_grammar_step',
    'run_lexicon_step',
    'run_translation_step',
    
    # High-level runner
    'run_conglanger',
]
