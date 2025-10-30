import os
import subprocess
from synthetic.config import CONGLANGER_PATH, OUTPUT_DIR, PROMPT_DIR
from synthetic.utils.logger import setup_logger


def run_conglanger(
    run_name=None,
    model="gemini-2.5-pro",
    steps=("phonology", "grammar", "lexicon", "translation"),
    custom_constraints=None,
    translation_sentence="Hello, world!",
    max_tokens=8192,
    temperature=0.7,
    top_p=None,
    thinking_budget=None,
    reasoning_effort="medium",
    sleep_between_calls=30,
    qa_enabled=True,
    self_refine_steps=3,
    qa_threshold=8.0,
    qa_thresholds_per_step=None,
    continue_qa=False,
    prompt_dir=PROMPT_DIR,
    output_dir=None,
    debug=False,
    extra_args=None,
):
    """Wrapper for running Conglanger's run_pipeline.py from within your repo."""
    
    # Logger setup
    logger = setup_logger("generate_language")

    # Resolve paths
    if not os.path.exists(CONGLANGER_PATH):
        raise FileNotFoundError(f"Conglanger entry point not found at: {CONGLANGER_PATH}")

    base_output = output_dir or OUTPUT_DIR
    os.makedirs(base_output, exist_ok=True)

    run_name = run_name or "unnamed_run"
    output_dir = os.path.join(base_output, run_name)
    os.makedirs(output_dir, exist_ok=True)

    # Build base command
    cmd = [
        "python", CONGLANGER_PATH,
        "--model", model,
        "--steps", ",".join(steps),
        "--output-dir", output_dir,
        "--max-tokens", str(max_tokens),
        "--temperature", str(temperature),
        "--sleep-between-calls", str(sleep_between_calls),
        "--prompt-dir", prompt_dir,
    ]

    if top_p is not None:
        cmd += ["--top_p", str(top_p)]
    if custom_constraints:
        cmd += ["--custom-constraints", custom_constraints]
    if translation_sentence:
        cmd += ["--translation-sentence", translation_sentence]
    if reasoning_effort:
        cmd += ["--reasoning-effort", reasoning_effort]
    if thinking_budget:
        cmd += ["--thinking-budget", str(thinking_budget)]

    # QA loop configuration
    if qa_enabled:
        cmd.append("--qa-enabled")
        cmd += ["--self-refine-steps", str(self_refine_steps)]

        if qa_threshold is not None:
            cmd += ["--qa-threshold", str(qa_threshold)]

        if qa_thresholds_per_step:
            for step, threshold in qa_thresholds_per_step.items():
                cmd += [f"--qa-threshold-{step}", str(threshold)]

        if continue_qa:
            cmd.append("--continue-qa")

    # Debug mode
    if debug:
        cmd.append("--debug")

    # 5. Add extra args
    if extra_args:
        cmd += extra_args

    # Run subprocess
    logger.info(f"Running Conglanger pipeline for run: {run_name}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Output directory: {output_dir}")

    try:
        subprocess.run(cmd, check=True)
        logger.info("Conglanger pipeline completed successfully.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Conglanger pipeline failed (exit code {e.returncode})")
        raise

    return output_dir
