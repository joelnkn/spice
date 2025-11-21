"""
Fine-tune language models using LoRA/QLoRA for multi-task learning.

This script supports fine-tuning on multiple NLP tasks including:
- NLI (Natural Language Inference)
- Sentiment Analysis
- Paraphrase Detection
- Translation

Usage Examples:
    # Basic training with default config
    python3 -m ft.train configs/train.yaml

    # Specify training dataset via command line (overrides config)
    python3 -m ft.train configs/train.yaml --train-path data/train_nli.jsonl

    # Train on XNLI data
    python3 -m ft.train configs/train.yaml --train-path data/xnli_train_16.jsonl

    # Train on multiple datasets (comma-separated)
    python3 -m ft.train configs/train.yaml --train-path data/train.jsonl,data/train_nli.jsonl

    # Using a custom config file
    python3 -m ft.train my_custom_config.yaml

    # Or specify dataset in config file (configs/train.yaml):
    #   train_path: data/train_nli.jsonl
    #   or
    #   train_path: [data/train.jsonl, data/train_nli.jsonl]

Configuration File (configs/train.yaml):
    The config file specifies:
    - model_name: Model to fine-tune (e.g., "google/mt5-base")
    - task_type: "seq2seq" or "causal"
    - use_qlora: true/false (requires CUDA if true)
    - train_path: Single file (str) or list of files for multi-task
    - eval_path: Validation/test file
    - Training hyperparameters (learning rate, batch size, etc.)

Data Format:
    Each line in your JSONL files should be:
    {
        "input": "Your input text here",
        "target": "Expected output",
        "task_id": "nli" | "sentiment" | "paraphrase" | "trans"
    }

    For NLI tasks, input format: "Premise: ... Hypothesis: ..."
    For other tasks, the input format is flexible and will be auto-formatted.

Output:
    Trained LoRA adapters are saved in the output directory:
    - outputs/run1/best_step{N}/ - Best checkpoint during training
    - outputs/run1/step{N}/ - Periodic checkpoints
    - outputs/run1/final/ - Final checkpoint after training
"""

import json, os, math, random
from dataclasses import dataclass
from typing import Dict, List, Optional
from omegaconf import OmegaConf

import torch
from accelerate import Accelerator
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from ft.utils import format_input_for_task

# --- helpers -----------------------------------------------------------------


def set_seed(seed: int):
    import numpy as np

    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def map_lora_targets(model, names):
    """Map generic names -> actual module names for common archs."""
    # Good defaults for T5/mT5 and BLOOM/Llama-like blocks.
    text = " ".join(n.lower() for n in names)
    if any(k in model.config.model_type for k in ["t5"]):
        return [
            "SelfAttention.q",
            "SelfAttention.k",
            "SelfAttention.v",
            "SelfAttention.o",
            "DenseReluDense.wi",
            "DenseReluDense.wo",
            "DenseReluDense.wi_0",
            "DenseReluDense.wi_1",
        ]
    # decoder-only (bloom/llama/mixtral/qwen)
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_tokenizer(model_name):
    # tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # mt5/t5 have no pad_token in some checkpoints -> align with eos
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def get_model(model_name, task_type, use_qlora):
    kwargs = {}
    if use_qlora:
        # 4-bit quantized backbone for QLoRA
        kwargs.update(
            dict(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                device_map="auto",
            )
        )
    if task_type == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model


def attach_lora(model, task_type, cfg):
    target_modules = map_lora_targets(model, cfg.lora.target_modules)
    peft_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        bias="none",
        task_type=(
            TaskType.SEQ_2_SEQ_LM if task_type == "seq2seq" else TaskType.CAUSAL_LM
        ),
        target_modules=target_modules,
    )
    return get_peft_model(model, peft_cfg)


# --- dataset & collator ------------------------------------------------------


def load_jsonl_dataset(path):
    # streamable local JSONL via datasets
    # Support both single file (str) and multiple files (list)
    # Handle OmegaConf ListConfig by converting to list
    if isinstance(path, str):
        return load_dataset("json", data_files=path, split="train")
    else:
        # Multiple files: concatenate them
        # Convert OmegaConf ListConfig or other iterables to Python list
        path_list = (
            OmegaConf.to_container(path, resolve=True)
            if hasattr(path, "__iter__") and not isinstance(path, str)
            else path
        )
        if not isinstance(path_list, list):
            path_list = [path_list]
        datasets = [
            load_dataset("json", data_files=p, split="train") for p in path_list
        ]
        from datasets import concatenate_datasets

        return concatenate_datasets(datasets)


@dataclass
class Collator:
    tokenizer: AutoTokenizer
    task_type: str
    max_src: int
    max_tgt: int

    def __call__(self, batch: List[Dict]):
        # Format inputs based on task
        inputs = [format_input_for_task(ex) for ex in batch]

        if self.task_type == "seq2seq":
            targets = [ex.get("target", "") for ex in batch]
            enc = self.tokenizer(
                inputs,
                max_length=self.max_src,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            with self.tokenizer.as_target_tokenizer():
                dec = self.tokenizer(
                    targets,
                    max_length=self.max_tgt,
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
            enc["labels"] = dec["input_ids"]
            return enc
        else:
            # causal: pack as "<prompt><answer><eos>" and shift by Trainer
            merged = [
                f"{inp}{self.tokenizer.eos_token}{ex.get('target','')}{self.tokenizer.eos_token}"
                for inp, ex in zip(inputs, batch)
            ]
            enc = self.tokenizer(
                merged,
                max_length=self.max_src + self.max_tgt + 4,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )
            enc["labels"] = enc["input_ids"].clone()
            return enc


# --- training loop (Accelerate) ----------------------------------------------


def main(cfg_path="configs/train.yaml", train_path=None, resume_from=None):
    """
    Main training function.

    Args:
        cfg_path: Path to config YAML file
        train_path: Optional path to training data file(s). Overrides config if provided.
                    Can be a single file (str) or list of files.
        resume_from: Optional path to existing checkpoint to resume training from.
                     If provided, loads existing LoRA adapters instead of creating new ones.
    """
    cfg = OmegaConf.load(cfg_path)

    # Override train_path if provided as argument
    if train_path is not None:
        cfg.io.train_path = train_path
        print(f"Using training data from command line: {train_path}")

    set_seed(cfg.seed)

    accelerator = Accelerator(gradient_accumulation_steps=cfg.train.grad_accum)
    is_main = accelerator.is_main_process
    if is_main:
        os.makedirs(cfg.io.out_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(cfg.io.out_dir, "resolved_config.yaml"))

    tokenizer = load_tokenizer(cfg.model_name)
    model = get_model(cfg.model_name, cfg.task_type, cfg.use_qlora)

    if cfg.use_qlora:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=cfg.train.gradient_checkpointing
        )

    if cfg.train.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Load existing LoRA adapters or create new ones
    if resume_from is not None:
        if is_main:
            accelerator.print(f"Resuming training from checkpoint: {resume_from}")
        model = PeftModel.from_pretrained(model, resume_from, is_trainable=True)
    else:
        model = attach_lora(model, cfg.task_type, cfg)

    train_ds = load_jsonl_dataset(cfg.io.train_path)
    eval_ds = load_jsonl_dataset(cfg.io.eval_path)

    collate = Collator(
        tokenizer, cfg.task_type, cfg.train.max_src_len, cfg.train.max_tgt_len
    )

    # Dataloaders
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.per_device_batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=max(1, cfg.train.per_device_batch_size * 2),
        shuffle=False,
        collate_fn=collate,
    )

    # Optim & sched
    from bitsandbytes import optim as bnb_optim

    if cfg.optim.name.lower().startswith("paged_adamw"):
        opt = bnb_optim.PagedAdamW8bit(
            model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
        )
    else:
        opt = torch.optim.AdamW(
            model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
        )

    # steps
    num_update_steps = cfg.train.max_steps
    warmup = int(num_update_steps * cfg.optim.warmup_ratio)
    sched = get_cosine_schedule_with_warmup(opt, warmup, num_update_steps)

    model, opt, train_loader, eval_loader, sched = accelerator.prepare(
        model, opt, train_loader, eval_loader, sched
    )

    # Logging
    step = 0
    best_eval = math.inf
    model.train()

    while step < cfg.train.max_steps:
        for batch in train_loader:
            with accelerator.accumulate(model):
                out = model(**{k: v.to(accelerator.device) for k, v in batch.items()})
                loss = out.loss
                accelerator.backward(loss)
                opt.step()
                sched.step()
                opt.zero_grad()
            if step % cfg.train.logging_steps == 0 and is_main:
                accelerator.print(f"step {step} | loss {loss.item():.4f}")
            if step % cfg.train.eval_steps == 0 and step > 0:
                eval_loss = evaluate_loop(model, eval_loader, accelerator)
                if is_main:
                    accelerator.print(f"[eval] step {step} | loss {eval_loss:.4f}")
                    if eval_loss < best_eval:
                        best_eval = eval_loss
                        save_adapters(
                            model, tokenizer, cfg.io.out_dir, f"best_step{step}"
                        )
            if step % cfg.train.save_steps == 0 and step > 0 and is_main:
                save_adapters(model, tokenizer, cfg.io.out_dir, f"step{step}")
            step += 1
            if step >= cfg.train.max_steps:
                break

    if is_main:
        save_adapters(model, tokenizer, cfg.io.out_dir, "final")


def evaluate_loop(model, loader, accelerator):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in loader:
            out = model(**{k: v.to(accelerator.device) for k, v in batch.items()})
            losses.append(accelerator.gather(out.loss.detach()).mean().item())
    model.train()
    return sum(losses) / max(1, len(losses))


def save_adapters(model, tokenizer, out_dir, tag):
    # Only PEFT adapter (small files) + tokenizer hash
    path = os.path.join(out_dir, tag)
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    with open(os.path.join(path, "card.json"), "w") as f:
        json.dump({"tokenizer_files": list(tokenizer.init_kwargs.keys())}, f)


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune language models with LoRA/QLoRA"
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="configs/train.yaml",
        help="Path to config YAML file (default: configs/train.yaml)",
    )
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Path to training data file(s). Can be a single file or comma-separated list. "
        "Overrides config file if provided. Example: --train-path data/train_nli.jsonl "
        "or --train-path data/train.jsonl,data/train_nli.jsonl",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to existing checkpoint to resume training from. "
        "Loads existing LoRA adapters instead of creating new ones. "
        "Example: --resume-from outputs/run1/best_step1000",
    )

    args = parser.parse_args()

    # Parse train_path if it's a comma-separated list
    train_path = args.train_path
    if train_path and "," in train_path:
        train_path = [p.strip() for p in train_path.split(",")]

    main(cfg_path=args.config, train_path=train_path, resume_from=args.resume_from)
