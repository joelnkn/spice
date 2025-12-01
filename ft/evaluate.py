"""
Evaluate fine-tuned or base mT5 models on test datasets.

This script loads a trained model checkpoint (or base model) and evaluates it on test data,
computing task-specific metrics like accuracy, per-class accuracy, etc.

Usage Examples:
    # Evaluate base model without fine-tuning
    python -m ft.evaluate --base-only
    
    # Evaluate base model with custom test file
    python -m ft.evaluate --base-only --test-path data/my_test.jsonl
    
    # Evaluate using default config (automatically finds latest checkpoint)
    python -m ft.evaluate
    
    # Evaluate specific checkpoint
    python -m ft.evaluate --model-path outputs/run1/best_step1000
    
    # Evaluate specific checkpoint on custom test file
    python -m ft.evaluate --model-path outputs/run1/final --test-path data/xnli_test.jsonl
    
    # Use custom config file
    python -m ft.evaluate --config configs/custom.yaml --base-only
    
    # The script will:
    # 1. Load the base model or trained LoRA adapters from the checkpoint
    # 2. Run predictions on all test examples
    # 3. Compute metrics by task (NLI, sentiment, paraphrase, translation)
    # 4. Print results and save to evaluation_results.json

Arguments:
    --model-path: Path to checkpoint directory (optional, auto-detects if not provided)
    --test-path: Path to test JSONL file (optional, uses eval_path from config if not provided)
    --base-only: Evaluate base model without loading LoRA adapters (default: False)
    --config: Path to config file (default: configs/train.yaml)

Output:
    - Prints accuracy metrics to console
    - Saves detailed results to outputs/evaluation_results.json
"""
import json
import os
from typing import Dict, List
from omegaconf import OmegaConf
import torch
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
from ft.utils import format_input_for_task


def load_model_and_tokenizer(model_path, base_model_name, task_type="seq2seq", use_base_only: bool = False):
    """Load model and tokenizer.

    If use_base_only is True, load only the base model (no LoRA adapters).
    Otherwise, load fine-tuned model with LoRA adapters from model_path.
    """
    if use_base_only:
        model_path = base_model_name

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if task_type == "seq2seq":
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    
    # Load LoRA adapters
    model = base_model if use_base_only else PeftModel.from_pretrained(base_model, model_path)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def predict(model, tokenizer, input_text, task_type="seq2seq", max_length=128):
    """Generate prediction from model."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    # Move inputs to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=3,
            early_stopping=True
        )
    
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return prediction.strip()


def evaluate_nli(predictions: List[str], labels: List[str]) -> Dict:
    """Evaluate NLI predictions."""
    correct = sum(1 for p, l in zip(predictions, labels) if p.lower() == l.lower())
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0
    
    # Per-class accuracy (using numeric labels)
    classes = ["entailment", "neutral", "contradiction"] 
    class_counts = {c: {"correct": 0, "total": 0} for c in classes}
    
    for pred, label in zip(predictions, labels):
        label_clean = label.strip()
        pred_clean = pred.strip()
        if label_clean in class_counts:
            class_counts[label_clean]["total"] += 1
            if pred_clean == label_clean:
                class_counts[label_clean]["correct"] += 1
    
    per_class_acc = {
        c: class_counts[c]["correct"] / class_counts[c]["total"]
        if class_counts[c]["total"] > 0 else 0.0
        for c in classes
    }
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_class_accuracy": per_class_acc
    }


def evaluate_sentiment(predictions: List[str], labels: List[str]) -> Dict:
    """Evaluate sentiment predictions."""
    correct = sum(1 for p, l in zip(predictions, labels) if p.lower() == l.lower())
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


def evaluate_paraphrase(predictions: List[str], labels: List[str]) -> Dict:
    """Evaluate paraphrase detection predictions."""
    # Normalize predictions and labels
    normalize = lambda x: "yes" if "yes" in x.lower() or "paraphrase" in x.lower() else "no"
    norm_preds = [normalize(p) for p in predictions]
    norm_labels = [normalize(l) for l in labels]
    
    correct = sum(1 for p, l in zip(norm_preds, norm_labels) if p == l)
    total = len(predictions)
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }


def evaluate_translation(predictions: List[str], labels: List[str]) -> Dict:
    """Evaluate translation (simple exact match for now)."""
    exact_match = sum(1 for p, l in zip(predictions, labels) if p.strip().lower() == l.strip().lower())
    total = len(predictions)
    
    return {
        "exact_match": exact_match / total if total > 0 else 0.0,
        "exact_matches": exact_match,
        "total": total
    }


def normalize_answer(s: str) -> str:
    """Normalize answer text for QA evaluation."""
    import re
    import string
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    # Handle empty predictions or ground truths
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # If no common tokens, F1 is 0
    if len(common_tokens) == 0:
        return 0.0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def evaluate_tydiqa(predictions: List[str], labels: List[str]) -> Dict:
    """Evaluate TyDiQA predictions using F1 metric."""
    f1_scores = []
    
    for pred, label in zip(predictions, labels):
        f1_scores.append(compute_f1(pred, label))
    
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    
    return {
        "f1": avg_f1,
        "total": len(predictions)
    }


def main(cfg_path="configs/train.yaml", checkpoint_path=None, test_path=None, use_base_only: bool = False):
    """Main evaluation function."""
    cfg = OmegaConf.load(cfg_path)
    
    if use_base_only:
        checkpoint_path = cfg.model_name
    else:
        # Determine checkpoint path
        if checkpoint_path is None:
            # Look for best checkpoint or final checkpoint
            out_dir = cfg.io.out_dir
            if os.path.exists(os.path.join(out_dir, "final")):
                checkpoint_path = os.path.join(out_dir, "final")
            elif os.path.exists(os.path.join(out_dir, "best_step2000")):
                checkpoint_path = os.path.join(out_dir, "best_step2000")
            else:
                # Find latest checkpoint
                checkpoints = [d for d in os.listdir(out_dir) if d.startswith("step") or d == "final"]
                if checkpoints:
                    checkpoint_path = os.path.join(out_dir, sorted(checkpoints)[-1])
                else:
                    raise ValueError(f"No checkpoint found in {out_dir}")
    
    print(f"Loading model from: {checkpoint_path}")
    model, tokenizer = load_model_and_tokenizer(
        checkpoint_path, 
        cfg.model_name, 
        cfg.task_type,
        use_base_only=use_base_only,
    )
    
    # Load test data
    if test_path is None:
        test_path = cfg.io.eval_path
    
    print(f"Loading test data from: {test_path}")
    test_ds = load_dataset("json", data_files=test_path, split="train")
    print(f"Test dataset size: {len(test_ds)} examples")
    
    # Run predictions
    predictions = []
    labels = []
    task_ids = []
    
    print("Running predictions...")
    for i, example in enumerate(tqdm(test_ds)):
        input_text = format_input_for_task(example)
        prediction = predict(model, tokenizer, input_text, cfg.task_type, cfg.train.max_tgt_len)
        predictions.append(prediction)
        labels.append(example.get("target", ""))
        task_ids.append(example.get("task_id", ""))
        
        # Debug
        if i % 20 == 0:
          match = "✓" if predictions[i].lower() == labels[i].lower() else "✗"
          print(f"{i+1}. {match} Predicted: '{predictions[i]}' | Expected: '{labels[i]}'")
    
    
    # Evaluate by task
    results = {}
    all_tasks = set(task_ids)
    
    for task in all_tasks:
        task_preds = [p for p, t in zip(predictions, task_ids) if t == task]
        task_labels = [l for l, t in zip(labels, task_ids) if t == task]
        
        print(f"\n=== Task: {task.upper()} ===")
        print(f"Examples: {len(task_preds)}")
        
        if task.lower() == "nli":
            task_results = evaluate_nli(task_preds, task_labels)
        elif task.lower() == "sentiment":
            task_results = evaluate_sentiment(task_preds, task_labels)
        elif task.lower() == "paraphrase":
            task_results = evaluate_paraphrase(task_preds, task_labels)
        elif task.lower() == "trans":
            task_results = evaluate_translation(task_preds, task_labels)
        elif task.lower() in ["tydiqa", "qa"]:
            task_results = evaluate_tydiqa(task_preds, task_labels)
        else:
            # Generic accuracy
            correct = sum(1 for p, l in zip(task_preds, task_labels) if p.lower() == l.lower())
            task_results = {
                "accuracy": correct / len(task_preds) if task_preds else 0.0,
                "correct": correct,
                "total": len(task_preds)
            }
        
        results[task] = task_results
        
        # Print metrics based on task type
        if "f1" in task_results:
            print(f"F1 Score: {task_results['f1']:.4f}")
        else:
            print(f"Accuracy: {task_results.get('accuracy', 0.0):.4f}")
        
        if "per_class_accuracy" in task_results:
            print("Per-class accuracy:")
            for cls, acc in task_results["per_class_accuracy"].items():
                print(f"  {cls}: {acc:.4f}")
    
    # Overall accuracy
    overall_correct = sum(1 for p, l in zip(predictions, labels) if p.lower() == l.lower())
    overall_accuracy = overall_correct / len(predictions) if predictions else 0.0
    
    print(f"\n=== OVERALL ===")
    print(f"Overall Accuracy: {overall_accuracy:.4f} ({overall_correct}/{len(predictions)})")
    
    # Save results
    results["overall"] = {
        "accuracy": overall_accuracy,
        "correct": overall_correct,
        "total": len(predictions)
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(cfg.io.out_dir, exist_ok=True)
    
    results_path = os.path.join(cfg.io.out_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned or base mT5 models on test datasets.")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to checkpoint directory (optional, auto-detects if not provided)"
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Path to test JSONL file (optional, uses eval_path from config if not provided)"
    )
    parser.add_argument(
        "--base-only",
        action="store_true",
        help="Evaluate base model without loading LoRA adapters"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to config file (default: configs/train.yaml)"
    )
    
    args = parser.parse_args()
    
    main(
        cfg_path=args.config,
        checkpoint_path=args.model_path,
        test_path=args.test_path,
        use_base_only=args.base_only
    )

