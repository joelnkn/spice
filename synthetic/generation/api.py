from synthetic.config import OUTPUT_DIR
from synthetic.generation.conglanger_client import run_conglanger
import os


def generate_consistent_language(corpus, output_dir=OUTPUT_DIR, run_name="consistent"):
    # # generate base language
    run_conglanger(
        steps=("phonology", "grammar", "lexicon"),
        qa_enabled=False,
        output_dir=output_dir,
        run_name=run_name,
        reasoning_effort="low",
        iteration=True,
    )

    last_id_file = os.path.join(output_dir, run_name, "LAST_LANGUAGE_ID")
    with open(last_id_file, "r", encoding="utf-8") as f:
        language_id = f.read().strip()

    # make consistent using corpus
    print("Stabilizing")
    for sample in corpus:
        run_conglanger(
            steps=("translation",),
            translation_sentence=sample,
            output_dir=output_dir,
            qa_enabled=True,
            lang_id=language_id,
            run_name=run_name,
            iteration=True,
        )


def generate_language(**kwargs):
    result = run_conglanger(**kwargs)
    return result


if __name__ == "__main__":
    # out = generate_language(
    #     run_name="smoketest_gemini",
    #     model="gemini-2.5-flash-lite",
    #     temperature=0.2,
    #     thinking_budget=0,               # zero “reasoning tokens” if supported
    #     reasoning_effort="low",
    #     sleep_between_calls=0,
    #     qa_enabled=False,                # disable QA loop entirely
    #     self_refine_steps=0,             # belt-and-suspenders
    # )
    out = generate_consistent_language(["Hello world!", "The quick brown fox"])
