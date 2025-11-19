"""
Extract k random examples from mteb/amazon_reviews_multi dataset.
Amazon Reviews Multi is a multilingual product review dataset with star ratings.

Usage Examples:
    # Extract 16 random English examples and save to file
    python3 -m ft.extract_sent --k 16 --language en --split train --output data/amazon_train_16.jsonl

    # Extract 16 random examples from each language and save to file
    python3 -m ft.extract_sent --k 16 --split train --output data/amazon_train_16.jsonl
    
    # Extract 32 examples with a fixed seed (reproducible)
    python3 -m ft.extract_sent --k 32 --seed 42 --output data/amazon_train_32.jsonl
    
    # Just print examples without saving (for testing)
    python3 -m ft.extract_sent --k 5

    # Extract all test examples
    python3 -m ft.extract_sent --split test --output data/amazon_test.jsonl

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

AMAZON_LANGUAGES = ["en", "de", "es", "fr", "ja", "zh"]

def extract_amazon_examples(k=None, language="all", split="train", seed=None):
    """
    Extract k random examples from mteb/amazon_reviews_multi dataset.
    
    Args:
        k: Number of examples to extract (None = all examples)
        language: Language code (default: "all" for all languages)
        split: Dataset split ("train", "validation", "test")
        seed: Random seed for reproducibility (optional, only used when k is specified)
    
    Returns:
        List of examples in the format expected by the training script
    """
    print(f"Loading Amazon Reviews Multi dataset (language={language}, split={split})...")
    
    # Load Amazon Reviews Multi dataset
    # Dataset has text (review), label (star rating 0-4 representing 1-5 stars)
    dataset = load_dataset("mteb/amazon_reviews_multi", language, split=split)
    
    print(f"Total examples in dataset: {len(dataset)}")
    
    # Map star ratings (0-4 in dataset) to 1-5 stars
    # 0 -> 1 star, 1 -> 2 stars, ..., 4 -> 5 stars
    
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
        review_text = example["text"]
        # Convert 0-4 label to 1-5 star rating
        stars = example["label"] + 1
        
        examples.append({
            "input": review_text,
            "target": str(stars),
            "task_id": "sentiment"
        })
    
    print(f"Extracted {len(examples)} examples")
    
    return examples


def main():
    parser = argparse.ArgumentParser(description="Extract random examples from mteb/amazon_reviews_multi dataset")
    parser.add_argument("--k", type=int, default=None, help="Number of examples to extract")
    parser.add_argument("--language", type=str, default="all", help="Language code (default: 'all' for all languages, or specify e.g., 'en', 'de', 'fr')")
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"], 
                       help="Dataset split")
    parser.add_argument("--output", type=str, default=None, 
                       help="Output file path (JSONL format). If not provided, prints to stdout.")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()

    languages = AMAZON_LANGUAGES if args.language == "all" else [args.language]

    examples = []
    for language in languages:
        examples.extend(extract_amazon_examples(
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

