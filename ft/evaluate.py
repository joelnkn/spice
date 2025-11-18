"""
Evaluate fine-tuned models on test datasets.

This script loads a trained model checkpoint and evaluates it on test data,
computing task-specific metrics like accuracy, per-class accuracy, etc.

Usage Examples:
    # Evaluate using default config (automatically finds latest checkpoint)
    python -m ft.evaluate
    
    # Evaluate specific checkpoint
    python -m ft.evaluate outputs/run1/best_step1000
    
    # Evaluate specific checkpoint on custom test file
    python -m ft.evaluate outputs/run1/final data/xnli_test.jsonl
    
    # The script will:
    # 1. Load the trained LoRA adapters from the checkpoint
    # 2. Run predictions on all test examples
    # 3. Compute metrics by task (NLI, sentiment, paraphrase, translation)
    # 4. Print results and save to evaluation_results.json

Arguments:
    checkpoint_path: Path to checkpoint directory (optional, auto-detects if not provided)
    test_path: Path to test JSONL file (optional, uses eval_path from config if not provided)

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


def load_model_and_tokenizer(model_path, base_model_name, task_type="seq2seq"):
    """Load fine-tuned model with LoRA adapters."""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if task_type == "seq2seq":
        base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    return model, tokenizer


def format_input_for_task(example: Dict) -> str:
    """Format input text based on task type (same as in train.py)."""
    task_id = example.get("task_id", "").lower()
    input_text = example.get("input", "")
    
    if task_id == "nli":
        return input_text
    elif task_id == "sentiment":
        return f"Sentiment analysis: {input_text}"
    elif task_id == "paraphrase":
        return f"Paraphrase detection: {input_text}"
    elif task_id == "trans":
        return input_text
    else:
        return input_text


def predict(model, tokenizer, input_text, task_type="seq2seq", max_length=128):
    """Generate prediction from model."""
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        if task_type == "seq2seq":
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                num_beams=3,
                early_stopping=True
            )
        else:
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
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
    
    # Per-class accuracy
    classes = ["entailment", "contradiction", "neutral"]
    class_counts = {c: {"correct": 0, "total": 0} for c in classes}
    
    for pred, label in zip(predictions, labels):
        label_lower = label.lower()
        pred_lower = pred.lower()
        if label_lower in class_counts:
            class_counts[label_lower]["total"] += 1
            if pred_lower == label_lower:
                class_counts[label_lower]["correct"] += 1
    
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


def main(cfg_path="configs/train.yaml", checkpoint_path=None, test_path=None):
    """Main evaluation function."""
    cfg = OmegaConf.load(cfg_path)
    
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
        cfg.task_type
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
    for example in tqdm(test_ds):
        input_text = format_input_for_task(example)
        prediction = predict(model, tokenizer, input_text, cfg.task_type, cfg.train.max_tgt_len)
        predictions.append(prediction)
        labels.append(example.get("target", ""))
        task_ids.append(example.get("task_id", ""))
    
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
        else:
            # Generic accuracy
            correct = sum(1 for p, l in zip(task_preds, task_labels) if p.lower() == l.lower())
            task_results = {
                "accuracy": correct / len(task_preds) if task_preds else 0.0,
                "correct": correct,
                "total": len(task_preds)
            }
        
        results[task] = task_results
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
    
    results_path = os.path.join(cfg.io.out_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    import sys
    
    checkpoint = sys.argv[1] if len(sys.argv) > 1 else None
    test_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    main(checkpoint_path=checkpoint, test_path=test_file)

