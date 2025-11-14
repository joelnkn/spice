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
        path_list = OmegaConf.to_container(path, resolve=True) if hasattr(path, '__iter__') and not isinstance(path, str) else path
        if not isinstance(path_list, list):
            path_list = [path_list]
        datasets = [load_dataset("json", data_files=p, split="train") for p in path_list]
        from datasets import concatenate_datasets
        return concatenate_datasets(datasets)


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


def main(cfg_path="configs/train.yaml"):
    cfg = OmegaConf.load(cfg_path)
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

    main(sys.argv[1] if len(sys.argv) > 1 else "configs/train.yaml")
