"""
Arguments:
    --dataset: Dataset name (supported: "xnli", "paws-x", "squad")
    --k: Number of examples to extract. Omit to extract all examples.
    --language: Language code (default: "all" - extracts from all languages). Specify a language code (e.g., "en", "de", "fr") to extract from a single language.
    --split: Dataset split - "train", "validation", or "test" (default: "train")
    --output: Output file path in JSONL format (optional, prints to stdout if not provided)
    --seed: Random seed for reproducibility (optional, only used when k is specified)

Usage Examples:
    # Extract 16 random English examples and save to file
    python3 -m ft.extract --dataset xnli --k 16 --language en --split train --output data/xnli_train_16.jsonl

    # Extract 16 random examples from each language and save to file
    python3 -m ft.extract --dataset xnli --k 16 --split train --output data/xnli_train_16.jsonl
    
    # Extract 32 examples with a fixed seed (reproducible)
    python3 -m ft.extract --dataset xnli --k 32 --seed 42 --output data/xnli_train_32.jsonl
    
    # Just print examples without saving (for testing)
    python3 -m ft.extract --dataset xnli --k 5

    # Extract all test examples
    python3 -m ft.extract --dataset xnli --split test --output data/xnli_test.jsonl
"""
from datasets import load_dataset
import json
import argparse
import random

DATASET_LANGUAGES = {
    "xnli": ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"],
    "paws-x": ["de", "en", "es", "fr", "ja", "ko", "zh"],
    "squad": ["plain_text"],
    "copenlu/answerable_tydiqa": [""]
}

DATASET_LABELS = {
    "xnli": {
        0: "entailment",
        1: "neutral",
        2: "contradiction"
    },
    "paws-x": {
        0: "not_duplicate",
        1: "duplicate",
    }
}

def format_xnli(row):
    premise = row["premise"]
    hypothesis = row["hypothesis"]
    label = DATASET_LABELS["xnli"].get(row["label"], "neutral")
    
    input_text = f"Premise: {premise} Hypothesis: {hypothesis}"
    
    return {
        "input": input_text,
        "target": label,
        "task_id": "nli"
    }

def format_paws_x(row):
    sentence1 = row["sentence1"]
    sentence2 = row["sentence2"]
    label = DATASET_LABELS["paws-x"][row["label"]]
    
    input_text = f"Sentence 1: {sentence1} Sentence 2: {sentence2}"
    
    return {
        "input": input_text,
        "target": label,
        "task_id": "paws-x"
    }
    
def format_squad(row):
    question = row["question"]
    context = row["context"]
    
    input_text = f"question: {question} context: {context}"
    label = row["answers"]["text"][0]
    return {
        "input": input_text,
        "target": label,
        "task_id": "squad"
    }

def format_tydiqa(row):
    question = row["question_text"]
    context = row["document_plaintext"]
    
    input_text = f"question: {question} context: {context}"
    label = row["annotations"]["answer_text"][0]
    return {
        "input": input_text,
        "target": label,
        "task_id": "tydiqa"
    }

DATASET_FORMAT = {
    "xnli": format_xnli,
    "paws-x": format_paws_x,
    "squad": format_squad,
    "copenlu/answerable_tydiqa": format_tydiqa,
}

def extract(dataset_name="xnli", k=None, language="all", split="train", seed=None):
    """
    Extract k random examples from dataset.
    
    Args:
        dataset_name: Dataset name in BUFFET (default: "xnli")
        k: Number of examples to extract (None = all examples)
        language: Language code (default: "all" for all languages)
        split: Dataset split ("train", "validation", "test")
        seed: Random seed for reproducibility (optional, only used when k is specified)
    
    Returns:
        List of examples in the format expected by the training script
    """
    languages = DATASET_LANGUAGES[dataset_name] if language == "all" else [language]
    format = DATASET_FORMAT[dataset_name]
    examples = []
    
    for language in languages:
        print(f"Loading {dataset_name} dataset (language={language}, split={split})...")

        dataset = load_dataset(dataset_name, language, split=split)
        
        print(f"Total examples in dataset: {len(dataset)}")

        if k is None:
            indices = range(len(dataset))
        else:
            if seed is not None:
                random.seed(seed)
            # Randomly sample k examples
            k = min(k, len(dataset))
            indices = random.sample(range(len(dataset)), k)
        
        for idx in indices:
            example = dataset[idx]
            examples.append(format(example))
        
        print(f"Extracted {len(examples)} examples")
    
    return examples


def main():
    parser = argparse.ArgumentParser(description="Extract random examples from XNLI dataset")
    parser.add_argument("--dataset", type=str, default="xnli", help="Dataset name in BUFFET (default: 'xnli')")
    parser.add_argument("--k", type=int, default=None, help="Number of examples to extract")
    parser.add_argument("--language", type=str, default="all", help="Language code (default: 'all' for all languages, or specify e.g., 'en', 'de', 'fr')")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"], 
                       help="Dataset split")
    parser.add_argument("--output", type=str, default=None, 
                       help="Output file path (JSONL format). If not provided, prints to stdout.")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()

    examples = extract(args.dataset, args.k, args.language, args.split, args.seed)

    # Save to file if output_path is provided
    if args.output:
        print(f"Saving to {args.output}...")
        with open(args.output, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + "\n")
        print(f"Saved {len(examples)} examples to {args.output}")
    
    if not args.output:
        # Print examples
        print("\nExtracted examples:")
        for i, ex in enumerate(examples, 1):
            print(f"\n{i}. {ex}")


if __name__ == "__main__":
    main()

