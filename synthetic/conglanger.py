"""Wrapper module for conglanger to enable clean imports."""

import os
import sys
from synthetic.config import CONGLANGER_PATH, OUTPUT_DIR, PROMPT_DIR
from synthetic.utils.logger import setup_logger

# Add conglanger to path
_conglanger_src = os.path.join(os.path.dirname(__file__), '../third_party/conglanger/src')
if _conglanger_src not in sys.path:
    sys.path.insert(0, _conglanger_src)

def run_conglanger(
    lang_name,
    run_name=None,
    random=False,
    iteration=0,
    model="gemini-2.5-flash", # prob should change back to gemini-2.5-pro
    steps=("affix", "lexicon"),
    custom_constraints=None,
    translation_sentence=None,
    max_tokens=8192,
    temperature=0.3,
    thinking_budget=0,
    reasoning_effort="low",
    sleep_between_calls=0,
    qa_enabled=True,
    self_refine_steps=4,
    qa_threshold=8.0,
    qa_thresholds_per_step=None,
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
        sys.executable,
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
        "--iteration",
        str(iteration),
        "--lang-name",
        str(lang_name)
    ]

    if custom_constraints:
        cmd += ["--custom-constraints", custom_constraints]
    if translation_sentence:
        cmd += ["--translation-sentence", translation_sentence]
    if reasoning_effort:
        cmd += ["--reasoning-effort", reasoning_effort]
    if thinking_budget:
        cmd += ["--thinking-budget", str(thinking_budget)]
    if lang_id:
        cmd += ["--lang-id", str(lang_id)]
    if random:
        cmd.append("--random")

    # QA loop configuration
    if qa_enabled:
        cmd.append("--qa-enabled")
        cmd += ["--self-refine-steps", str(self_refine_steps)]

        if qa_threshold is not None:
            cmd += ["--qa-threshold", str(qa_threshold)]

        # Set per-step thresholds if not provided
        if qa_thresholds_per_step is None:
            qa_thresholds_per_step = {"lexicon": 9, "affix": 9}
        
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
    # High-level runner
    'run_conglanger',
]
