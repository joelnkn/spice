import os
import subprocess
from synthetic.config import CONGLANGER_PATH, OUTPUT_DIR, PROMPT_DIR
from synthetic.utils.logger import setup_logger
from synthetic.analysis.features import extract_and_save_from_analysis


# Uses same defaults as Conlanger
def run_conglanger(
    run_name=None,
    model="gemini-2.5-pro",
    steps=("phonology", "grammar", "lexicon", "translation"),
    custom_constraints=None,
    translation_sentence="Hello, world!",
    max_tokens=8192,
    temperature=0.7,
    thinking_budget=1000,
    reasoning_effort="medium",
    sleep_between_calls=30,
    qa_enabled=True,
    self_refine_steps=3,
    qa_threshold=8.0,
    qa_thresholds_per_step=None,
    prompt_dir=PROMPT_DIR,
    output_dir=OUTPUT_DIR,
    lang_id=None,
    debug=False,
    run_analysis=True,
    extra_args=None,
):
    """Wrapper for running Conglanger's run_pipeline.py from within your repo."""

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

    # Analysis
    if run_analysis:
        cmd.append("--run-analysis")

    # Add extra args
    if extra_args:
        cmd += extra_args

    # Run subprocess
    logger.info(f"Running Conglanger pipeline for run: {run_name}")
    logger.info(f"Command: {' '.join(cmd)}")
    logger.info(f"Output directory: {output_dir}")

    try:
        subprocess.run(cmd, check=True)
        logger.info("Conglanger pipeline completed successfully.")
        # If run_analysis was requested, run feature extraction on produced languages
        if run_analysis:
            # Only run feature extraction for the language produced by this run.
            languages_dir = os.path.join(output_dir, "languages")
            target_lang = None

            last_id_file = os.path.join(output_dir, "LAST_LANGUAGE_ID")
            if os.path.exists(last_id_file):
                try:
                    with open(last_id_file, "r", encoding="utf-8") as f:
                        lid_val = f.read().strip()
                    candidate = os.path.join(output_dir, "languages", lid_val)
                    if os.path.exists(candidate):
                        target_lang = candidate
                except Exception:
                    pass

            if target_lang:
                lid = os.path.basename(target_lang)
                try:
                    logger.info(
                        f"Running feature extraction for generated language: {lid}"
                    )
                    res = extract_and_save_from_analysis(target_lang)
                    if res:
                        logger.info(f"Feature extraction saved: {res}")
                    else:
                        logger.warning(
                            f"No analysis found for language {lid}; skipping feature extraction"
                        )
                except Exception as e:
                    logger.error(f"Feature extraction failed for {lid}: {e}")
            else:
                logger.warning(
                    f"Could not locate generated language folder under {output_dir}; skipping feature extraction"
                )
    except subprocess.CalledProcessError as e:
        logger.error(f"Conglanger pipeline failed (exit code {e.returncode})")
        raise

    return output_dir
