"""
Extract k random examples from XNLI dataset.
XNLI is a cross-lingual natural language inference dataset.

Usage Examples:
    # Extract 16 random English examples and save to file
    python3 -m ft.extract_xnli --k 16 --language en --split train --output data/xnli_train_16.jsonl

    # Extract 16 random examples from each language and save to file
    python3 -m ft.extract_xnli --k 16 --split train --output data/xnli_train_16.jsonl
    
    # Extract 32 examples with a fixed seed (reproducible)
    python3 -m ft.extract_xnli --k 32 --seed 42 --output data/xnli_train_32.jsonl
    
    # Just print examples without saving (for testing)
    python3 -m ft.extract_xnli --k 5

    # Extract all test examples
    python3 -m ft.extract_xnli --split test --output data/xnli_test.jsonl

Arguments:
    --k: Number of examples to extract. Omit to extract all examples.
    --language: Language code (default: "all" - extracts from all languages). Specify a language code (e.g., "en", "de", "fr") to extract from a single language.
    --split: Dataset split - "train", "validation", or "test" (default: "train")
    --output: Output file path in JSONL format (optional, prints to stdout if not provided)
    --seed: Random seed for reproducibility (optional, only used when k is specified)
"""
from datasets import load_dataset
import json
import argparse
import random

XNLI_LANGUAGES = ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"]

def extract_xnli_examples(k=None, language="all", split="train", seed=None):
    """
    Extract k random examples from XNLI dataset.
    
    Args:
        k: Number of examples to extract (None = all examples)
        language: Language code (default: "all" for all languages)
        split: Dataset split ("train", "validation", "test")
        seed: Random seed for reproducibility (optional, only used when k is specified)
    
    Returns:
        List of examples in the format expected by the training script
    """
    print(f"Loading XNLI dataset (language={language}, split={split})...")
    
    # Load XNLI dataset
    # XNLI has premise, hypothesis, and label fields
    dataset = load_dataset("xnli", language, split=split)
    
    print(f"Total examples in dataset: {len(dataset)}")
    
    # Map XNLI labels to our format (using numbers for simpler generation)
    label_map = {
        0: "entailment",  # entailment
        1: "neutral",  # neutral
        2: "contradiction"   # contradiction
    }
    
    if k is None:
        indices = range(len(dataset))
    else:
        if seed is not None:
            random.seed(seed)
        # Randomly sample k examples
        k = min(k, len(dataset))
        indices = random.sample(range(len(dataset)), k)
    
    examples = []
    for idx in indices:
        example = dataset[idx]
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        label = label_map.get(example["label"], "neutral")
        
        # Format as our NLI format: "Premise: ... Hypothesis: ..."
        input_text = f"Premise: {premise} Hypothesis: {hypothesis}"
        
        examples.append({
            "input": input_text,
            "target": label,
            "task_id": "nli"
        })
    
    print(f"Extracted {len(examples)} examples")
    
    return examples


def main():
    parser = argparse.ArgumentParser(description="Extract random examples from XNLI dataset")
    parser.add_argument("--k", type=int, default=None, help="Number of examples to extract")
    parser.add_argument("--language", type=str, default="all", help="Language code (default: 'all' for all languages, or specify e.g., 'en', 'de', 'fr')")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"], 
                       help="Dataset split")
    parser.add_argument("--output", type=str, default=None, 
                       help="Output file path (JSONL format). If not provided, prints to stdout.")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()

    languages = XNLI_LANGUAGES if args.language == "all" else [args.language]

    examples = []
    for language in languages:
        examples.extend(extract_xnli_examples(
        k=args.k,
        language=language,
        split=args.split,
        seed=args.seed)
    )

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

