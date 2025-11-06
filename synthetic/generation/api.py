from synthetic.generation.conglanger_client import run_conglanger

def generate_language(**kwargs):
    result = run_conglanger(**kwargs)
    return result

if __name__ == "__main__":
    out = generate_language(
        run_name="smoketest_gemini",
        model="gemini-2.5-flash-lite",   
        temperature=0.2,
        thinking_budget=0,               # zero “reasoning tokens” if supported
        reasoning_effort="low",
        sleep_between_calls=0,
        qa_enabled=False,                # disable QA loop entirely
        self_refine_steps=0,             # belt-and-suspenders
    )