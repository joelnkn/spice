from typing import Dict


def format_input_for_task(example: Dict) -> str:
    """Format input text based on task type."""
    task_id = example.get("task_id", "").lower()
    input_text = example.get("input", "")

    if task_id == "nli":
        # NLI: already formatted as "Premise: ... Hypothesis: ..."
        return input_text
    elif task_id == "sentiment":
        # Sentiment: format as "Sentiment analysis: <text>"
        return f"Sentiment analysis: {input_text}"
    elif task_id == "paraphrase":
        # Paraphrase: format as "Paraphrase detection: <sentence1> | <sentence2>"
        return f"Paraphrase detection: {input_text}"
    elif task_id == "trans":
        # Translation: keep as is
        return input_text
    else:
        # Default: use input as is
        return input_text
