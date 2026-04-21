"""
Real-Data CL Experiment — "Stop Tuning the Dynamics"

Unified script for Qwen3.5-0.8B, Jamba-Reasoning-3B, Nemotron-3-Nano-4B.
Uses REAL HuggingFace datasets, not synthetic templates.

Usage:
  python3 run_real_data.py --model qwen     # Qwen3.5-0.8B
  python3 run_real_data.py --model jamba     # AI21-Jamba-Reasoning-3B
  python3 run_real_data.py --model nemotron  # NVIDIA-Nemotron-3-Nano-4B

Datasets:
  Code:         Nan-Do/code-search-net-python (field: code)
  Science:      ccdv/pubmed-summarization (field: abstract)
  Math:         openai/gsm8k (field: question+answer)
  Conversation: HuggingFaceH4/ultrachat_200k (field: messages)

Output: /home/ubuntu/results/<model_name>/
"""

import time
import json
import math
import random
import gc
import os
import sys
import subprocess
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import LoraConfig, get_peft_model, TaskType


# ============================================================
# Model configs
# ============================================================

MODEL_CONFIGS = {
    "qwen": {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "attn_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "ssm_targets": ["in_proj", "out_proj", "gate_proj"],
        "ssm_label": "deltanet",
        "trust_remote_code": True,
        "extra_kwargs": {},
    },
    "jamba": {
        "model_id": "ai21labs/AI21-Jamba-Reasoning-3B",
        "attn_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "ssm_targets": ["in_proj", "out_proj"],
        "ssm_label": "mamba",
        "trust_remote_code": False,
        "extra_kwargs": {"use_mamba_kernels": False},  # avoid mamba-ssm dependency
    },
    "nemotron": {
        "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "attn_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "ssm_targets": ["in_proj", "out_proj"],
        "ssm_label": "mamba2",
        "trust_remote_code": True,
        "extra_kwargs": {},
    },
}

# ============================================================
# Dataset configs — ALL REAL DATA
# ============================================================

DATASET_CONFIGS = {
    "code": {
        "name": "Nan-Do/code-search-net-python",
        "config": None,
        "split": "train",
        "text_fn": lambda x: x["code"],
        "min_words": 30,
        "license": "MIT (CodeSearchNet)",
    },
    "science": {
        "name": "ccdv/pubmed-summarization",
        "config": "document",
        "split": "train",
        "text_fn": lambda x: x["abstract"],
        "min_words": 50,
        "license": "Public domain (PubMed)",
    },
    "math": {
        "name": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "text_fn": lambda x: "Question: " + x["question"] + "\nAnswer: " + x["answer"],
        "min_words": 30,
        "license": "MIT",
    },
    "conversation": {
        "name": "HuggingFaceH4/ultrachat_200k",
        "config": None,
        "split": "train_sft",
        "text_fn": lambda x: " ".join(m["content"] for m in x["messages"]),
        "min_words": 50,
        "license": "MIT",
    },
}

# ============================================================
# Experiment config
# ============================================================

NUM_TRAIN = 2000
NUM_TEST = 400
NUM_EPOCHS = 3
LR = 2e-4
LORA_RANK = 16
LORA_ALPHA = 32
MAX_LEN = 512
SEED = 42
STRATEGIES = ["lora_attention", "lora_ssm", "all_linear"]


def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Data loading
# ============================================================

def load_real_datasets(tokenizer, seed=SEED):
    """Load real data from HuggingFace. Returns dict of domain -> {train, test}."""
    rng = random.Random(seed)
    tasks = {}

    for domain, cfg in DATASET_CONFIGS.items():
        print(f"  Loading {domain}: {cfg['name']}...")
        if cfg["config"]:
            ds = load_dataset(cfg["name"], cfg["config"], split=cfg["split"], streaming=True)
        else:
            ds = load_dataset(cfg["name"], split=cfg["split"], streaming=True)

        # Collect examples, filtering short ones
        texts = []
        for item in ds:
            if len(texts) >= NUM_TRAIN + NUM_TEST + 100:
                break
            text = cfg["text_fn"](item)
            if len(text.split()) < cfg["min_words"]:
                continue
            # Tokenize to verify it works
            toks = tokenizer(text, max_length=MAX_LEN, truncation=True)
            if len(toks["input_ids"]) < 20:  # skip very short after tokenization
                continue
            texts.append(text)

        rng.shuffle(texts)
        tasks[domain] = {
            "train": texts[:NUM_TRAIN],
            "test": texts[NUM_TRAIN:NUM_TRAIN + NUM_TEST],
        }
        print(f"    Collected: {len(texts)} total, {len(tasks[domain]['train'])} train, {len(tasks[domain]['test'])} test")

    return tasks


# ============================================================
# LoRA target modules
# ============================================================

def get_target_modules(strategy, model_cfg):
    if strategy == "lora_attention":
        return model_cfg["attn_targets"]
    elif strategy == "lora_ssm":
        return model_cfg["ssm_targets"]
    elif strategy == "all_linear":
        return "all-linear"
    return "all-linear"


# ============================================================
# Training and evaluation
# ============================================================

def train_epoch(model, tokenizer, texts, optimizer, device):
    model.train()
    total_loss, n = 0.0, 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=MAX_LEN,
                           truncation=True, padding="max_length").to(device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def eval_ppl(model, tokenizer, texts, device):
    model.eval()
    total_loss, n = 0.0, 0
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=MAX_LEN,
                           truncation=True, padding="max_length").to(device)
        outputs = model(**inputs, labels=inputs["input_ids"])
        total_loss += outputs.loss.item()
        n += 1
    avg_loss = total_loss / max(n, 1)
    return math.exp(min(avg_loss, 20))  # cap to avoid overflow


# ============================================================
# CL experiment
# ============================================================

def run_cl_experiment(strategy, model_cfg, tasks, tokenizer, device, output_dir):
    domain_order = ["code", "science", "conversation", "math"]

    print(f"\n{'='*60}")
    print(f"CL: {strategy} | model={model_cfg['model_id']}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Load model
    load_kwargs = {"dtype": torch.bfloat16, "trust_remote_code": model_cfg["trust_remote_code"]}
    if model_cfg["extra_kwargs"]:
        load_kwargs.update(model_cfg["extra_kwargs"])

    model = AutoModelForCausalLM.from_pretrained(model_cfg["model_id"], **load_kwargs).to(device)

    # Apply LoRA
    target_modules = get_target_modules(strategy, model_cfg)
    print(f"  Targets: {target_modules}")

    lora_config = LoraConfig(
        r=LORA_RANK, lora_alpha=LORA_ALPHA,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM, lora_dropout=0.0,
    )
    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable:,} trainable / {total:,} total ({trainable/total*100:.2f}%)")

    # Baseline PPL
    ppl_matrix = []
    baseline = {}
    for d in domain_order:
        baseline[d] = eval_ppl(model, tokenizer, tasks[d]["test"][:50], device)  # eval on 50 for speed
    ppl_matrix.append(dict(baseline))
    print(f"  Baseline PPL: { {k: f'{v:.1f}' for k,v in baseline.items()} }")

    # Sequential training
    train_log = []
    for d in domain_order:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=LR
        )
        t0 = time.perf_counter()
        losses = []
        for epoch in range(NUM_EPOCHS):
            loss = train_epoch(model, tokenizer, tasks[d]["train"], optimizer, device)
            losses.append(loss)
        elapsed = time.perf_counter() - t0
        train_log.append({"task": d, "losses": losses, "time": elapsed})
        print(f"  {d}: {losses[0]:.3f} -> {losses[-1]:.3f} ({elapsed:.1f}s)")

        # Eval all domains
        step_ppl = {}
        for ed in domain_order:
            step_ppl[ed] = eval_ppl(model, tokenizer, tasks[ed]["test"][:50], device)
        ppl_matrix.append(step_ppl)

    # Save adapter
    adapter_dir = os.path.join(output_dir, "adapters", f"{strategy}_seed{SEED}")
    model.save_pretrained(adapter_dir)

    # Metrics
    forgetting = {}
    for d in domain_order:
        ppls = [ppl_matrix[s][d] for s in range(len(ppl_matrix))]
        best = min(ppls[1:])
        final = ppls[-1]
        forgetting[d] = {"best": float(best), "final": float(final), "delta": float(final - best)}

    result = {
        "strategy": strategy,
        "model": model_cfg["model_id"],
        "ssm_type": model_cfg["ssm_label"],
        "seed": SEED,
        "lora_rank": LORA_RANK,
        "num_train": NUM_TRAIN,
        "num_test": NUM_TEST,
        "max_len": MAX_LEN,
        "trainable_params": trainable,
        "total_params": total,
        "param_pct": f"{trainable/total*100:.2f}%",
        "baseline_ppl": {k: float(v) for k, v in baseline.items()},
        "ppl_matrix": [{k: float(v) for k, v in step.items()} for step in ppl_matrix],
        "forgetting": forgetting,
        "avg_forgetting": float(np.mean([v["delta"] for v in forgetting.values()])),
        "avg_final_ppl": float(np.mean([ppl_matrix[-1][d] for d in domain_order])),
        "train_log": train_log,
        "adapter_path": adapter_dir,
    }

    del model, optimizer
    clear_mem()
    return result


# ============================================================
# MMLU
# ============================================================

def run_mmlu(model_id, adapter_path, label, device, output_dir, trust_remote_code=False):
    """Run MMLU via lm-eval."""
    print(f"  MMLU [{label}]...")
    mmlu_dir = os.path.join(output_dir, "mmlu")
    os.makedirs(mmlu_dir, exist_ok=True)

    model_args = f"pretrained={model_id},dtype=bfloat16,trust_remote_code={trust_remote_code}"
    if adapter_path:
        model_args += f",peft={adapter_path}"

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", model_args,
        "--tasks", "mmlu",
        "--num_fewshot", "5",
        "--batch_size", "auto",
        "--device", device,
        "--output_path", mmlu_dir,
        "--limit", "100",
    ]

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.perf_counter() - t0

    result = {"label": label, "time": elapsed, "returncode": proc.returncode}
    if proc.returncode == 0:
        for line in proc.stdout.split("\n"):
            if "mmlu" in line.lower() and "acc" in line.lower() and "|" in line:
                result["raw_line"] = line.strip()
        result["stdout_tail"] = proc.stdout[-3000:]
    else:
        result["stderr_tail"] = proc.stderr[-2000:]
        print(f"    FAILED: {proc.stderr[-300:]}")

    clear_mem()
    print(f"    Done [{label}] in {elapsed:.1f}s | {result.get('raw_line', 'FAILED')[:60]}")
    return result


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen", "jamba", "nemotron"])
    parser.add_argument("--skip-mmlu", action="store_true")
    args = parser.parse_args()

    model_cfg = MODEL_CONFIGS[args.model]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = f"/home/ubuntu/results/{args.model}"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print(f"REAL-DATA CL EXPERIMENT: {model_cfg['model_id']}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Train: {NUM_TRAIN}/domain | Test: {NUM_TEST}/domain | MaxLen: {MAX_LEN}")
    print(f"LoRA: r={LORA_RANK}, alpha={LORA_ALPHA}")
    print(f"Strategies: {STRATEGIES}")
    print("=" * 70)

    t_start = time.perf_counter()

    # Load tokenizer and data
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_id"], trust_remote_code=model_cfg["trust_remote_code"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\nLoading datasets...")
    tasks = load_real_datasets(tokenizer, seed=SEED)

    # Save dataset info
    ds_info = {}
    for domain, cfg in DATASET_CONFIGS.items():
        ds_info[domain] = {
            "dataset": cfg["name"],
            "config": cfg["config"],
            "license": cfg["license"],
            "train_size": len(tasks[domain]["train"]),
            "test_size": len(tasks[domain]["test"]),
        }
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(ds_info, f, indent=2)

    # Run CL experiments
    all_results = {"model": model_cfg, "datasets": ds_info, "cl_results": []}

    for strategy in STRATEGIES:
        result = run_cl_experiment(strategy, model_cfg, tasks, tokenizer, device, output_dir)
        all_results["cl_results"].append(result)
        # Save incrementally
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Print CL summary
    print(f"\n{'='*70}")
    print("CL SUMMARY")
    print(f"{'='*70}")
    for r in all_results["cl_results"]:
        print(f"  {r['strategy']:20s} params={r['param_pct']:>6s}  PPL={r['avg_final_ppl']:>10.1f}  forget={r['avg_forgetting']:>10.1f}")

    # MMLU
    if not args.skip_mmlu:
        print(f"\n{'='*70}")
        print("MMLU EVALUATION")
        print(f"{'='*70}")

        mmlu_results = {}
        # Base
        mmlu_results["base"] = run_mmlu(
            model_cfg["model_id"], None, "base", device, output_dir,
            trust_remote_code=model_cfg["trust_remote_code"]
        )
        # Each adapter
        for r in all_results["cl_results"]:
            label = f"{r['strategy']}_seed{SEED}"
            mmlu_results[label] = run_mmlu(
                model_cfg["model_id"], r["adapter_path"], label, device, output_dir,
                trust_remote_code=model_cfg["trust_remote_code"]
            )

        all_results["mmlu_results"] = mmlu_results

    # Final save
    total_time = time.perf_counter() - t_start
    all_results["meta"] = {
        "total_time_seconds": total_time,
        "total_time_human": f"{total_time/60:.1f} minutes",
        "device": device,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else "N/A",
        "torch_version": torch.__version__,
        "transformers_version": __import__("transformers").__version__,
        "peft_version": __import__("peft").__version__,
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save env
    subprocess.run(f"{sys.executable} -m pip freeze > {output_dir}/pip_freeze.txt", shell=True)
    subprocess.run(f"nvidia-smi > {output_dir}/nvidia_smi.txt 2>&1", shell=True)

    print(f"\n{'='*70}")
    print(f"DONE in {total_time/60:.1f} minutes")
    print(f"Results: {output_dir}/results.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
