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
import enum
from pydoc import text
from datasets import load_dataset, Dataset
import json
import argparse
import random
from collections import defaultdict

DATASET_LANGUAGES = {
    "xnli": ["ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru", "sw", "th", "tr", "ur", "vi", "zh"],
    "paws-x": ["de", "en", "es", "fr", "ja", "ko", "zh"],
    "squad": ["plain_text"],
    "copenlu/answerable_tydiqa": [""],
    "amazon": ["en"],
    "indic": ["ur", "te", "ta", "pa", "or", "mr", "ml", "kn", "hi", "gu", "bn", "bd", "as"],
    "xnli-conlang": [""],
}

def extract_amazon(language, split):
    assert split in ["train", "validation", "test"]
    return load_dataset(
        "json",
        data_files=f"hf://datasets/mteb/amazon_reviews_multi/en/{split}.jsonl",
    )["train"].filter(lambda row: row['label'] != 3)
    

def extract_indic(language, split):
    return load_dataset("mteb/IndicSentiment", language, split=split).filter(lambda row: row["LABEL"] is not None)


def extract_conlang_xnli(json_path: str, split="train", balance=False) -> Dataset:
    with open(json_path, "r", encoding="utf-8") as f:
        sentences = json.load(f)["sentences"]
    
    xnli = load_dataset("xnli", "en", split="train")

    xnli_dict = defaultdict(lambda: defaultdict(int))
    for i in range(1000):
        premise, hypothesis = xnli[i]["premise"], xnli[i]["hypothesis"]
        xnli_dict[premise][hypothesis] = i
    
    examples = []
    for i in range(len(sentences) - 1):
        s1, s2 = sentences[i], sentences[i + 1]
        # valid pair, keep
        if s1['global_index'] % 2 == 0 and s2['global_index'] == s1['global_index'] + 1:
            premise, hypothesis = s1['conlang_sentence'], s2['conlang_sentence']
            input_text = f"Premise: {premise} Hypothesis: {hypothesis}"
            # Use English sentences to look up the correct XNLI index
            eng_premise, eng_hypothesis = s1['english_sentence'], s2['english_sentence']
            xnli_index = xnli_dict[eng_premise][eng_hypothesis]
            label = DATASET_LABELS["xnli"].get(xnli[xnli_index]["label"], "neutral")

            if xnli[xnli_index]['premise'] != eng_premise or xnli[xnli_index]['hypothesis'] != eng_hypothesis:
                print(f"Warning: English sentence mismatch at index {xnli_index}")
                print(f"  JSON: {premise} {hypothesis}")
                print(f"  XNLI: {xnli[xnli_index]['premise']} {xnli[xnli_index]['hypothesis']}")
            
            example = {
                "input": input_text,
                "target": label,
                "task_id": "nli"
            }
            examples.append(example)
    
    # Balance dataset to 50 examples per label if flag is set
    if balance:
        label_examples = defaultdict(list)
        for ex in examples:
            label_examples[ex["target"]].append(ex)
        
        num_examples = min(len(label_examples[label]) for label in ["entailment", "neutral", "contradiction"])
        
        balanced_examples = []
        for label in ["entailment", "neutral", "contradiction"]:
            label_exs = label_examples[label][:num_examples]  # Take first 50 of each label
            balanced_examples.extend(label_exs)
            print(f"Label '{label}': {len(label_exs)} examples (requested 50)")
        
        # Shuffle to mix labels
        random.shuffle(balanced_examples)
        examples = balanced_examples
    
    return examples


    # # group by k = global_index // 2, using parity for role
    # pairs = {}
    # for s in sentences:
    #     gi = s["global_index"]
    #     k = gi // 2
    #     role = "premise" if gi % 2 == 0 else "hypothesis"
    #     pairs.setdefault(k, {})[role] = s["conlang_sentence"]

    # # keep only complete pairs, sorted by original k
    # ks = sorted(k for k, v in pairs.items() if "premise" in v and "hypothesis" in v)

    # xnli = load_dataset("xnli", "en", split="train")

    # examples = []
    # for k in ks:
    #     premise = pairs[k]["premise"]
    #     hypothesis = pairs[k]["hypothesis"]
    #     input_text = f"Premise: {premise} Hypothesis: {hypothesis}"
    #     label = DATASET_LABELS["xnli"].get(xnli[k]["label"], "neutral")
        
    #     eng_json = s["english_sentence"]
    #     eng_xnli = xnli[k][role]
        
    #     if eng_json != eng_xnli:
    #         print(f"Warning: English sentence mismatch at index {k}")
    #         print(f"  JSON: {eng_json}")
    #         print(f"  XNLI: {eng_xnli}")
        
    #     example = {
    #         "input": input_text,
    #         "target": label,
    #         "task_id": "nli"
    #     }
    #     examples.append(example)
        
    # if split == "train":
    #     return examples[:int(len(examples)*0.9)]
    # else:
    #     return examples[int(len(examples)*0.9):]


CUSTOM_EXTRACT = {
    "amazon": extract_amazon,
    "indic": extract_indic,
    "xnli-conlang": extract_conlang_xnli,
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
    
def format_amazon(row):
    input_text = row["text"]
    target = "Positive" if row["label"] > 3 else "Negative"
    return {
        "input": input_text,
        "target": target,
        "task_id": "sentiment"
    }
    
def format_indic(row):
    input_text = row["INDIC REVIEW"]
    target = row["LABEL"]
    return {
        "input": input_text,
        "target": target,
        "task_id": "sentiment"
    }

DATASET_FORMAT = {
    "xnli": format_xnli,
    "paws-x": format_paws_x,
    "squad": format_squad,
    "copenlu/answerable_tydiqa": format_tydiqa,
    "amazon": format_amazon,
    "indic": format_indic,
    "xnli-conlang": lambda x: x,
}

def extract(dataset_name="xnli", k=None, language="all", split="train", seed=None, balance=False):
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

        if dataset_name in CUSTOM_EXTRACT:
            if dataset_name == "xnli-conlang":
                dataset = CUSTOM_EXTRACT[dataset_name](language, split=split, balance=balance)
            else:
                dataset = CUSTOM_EXTRACT[dataset_name](language, split=split)
        else:
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
    parser.add_argument("--balance", action="store_true",
                       help="Balance dataset to 50 examples per label (only for xnli-conlang)")
    
    args = parser.parse_args()

    examples = extract(args.dataset, args.k, args.language, args.split, args.seed, args.balance)

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

