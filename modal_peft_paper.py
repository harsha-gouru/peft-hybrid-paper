"""
Modal runner for PEFT Hybrid CL Paper experiments.

Runs all 3 models (Qwen, Nemotron, Jamba) with real HuggingFace datasets,
multiple seeds, and MMLU evaluation on H100.

Usage:
    # Qwen 0.8B (4 strategies x 3 seeds, ~30 min, ~$2)
    modal run modal_peft_paper.py --model qwen

    # Nemotron 4B (3 strategies x 3 seeds, ~60 min, ~$4)
    modal run modal_peft_paper.py --model nemotron

    # Jamba 3B (4 strategies x 3 seeds, ~90 min, ~$6)
    modal run modal_peft_paper.py --model jamba

    # Quick smoke test
    modal run modal_peft_paper.py --model qwen --smoke

    # Skip MMLU (CL experiments only)
    modal run modal_peft_paper.py --model qwen --skip-mmlu

    # Custom seeds
    modal run modal_peft_paper.py --model qwen --seeds "42,123,7"

Datasets (real HuggingFace):
    Code:         Nan-Do/code-search-net-python
    Science:      ccdv/pubmed-summarization
    Math:         openai/gsm8k
    Conversation: HuggingFaceH4/ultrachat_200k
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

app = modal.App("peft-paper")

BASE_PACKAGES = [
    "torch",
    "transformers>=4.52",
    "peft>=0.7",
    "accelerate",
    "datasets",
    "numpy",
    "sentencepiece",
    "huggingface-hub",
    "lm-eval>=0.4",
    "packaging",
]

# Base image for Qwen and Jamba (no mamba-ssm needed)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*BASE_PACKAGES)
)

# CUDA devel image for Nemotron (needs mamba-ssm compiled from source)
nemotron_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.11"
    )
    .run_commands("pip install --upgrade pip setuptools wheel")
    .pip_install(*BASE_PACKAGES)
    .pip_install("ninja")
    .run_commands(
        # Only compile for H100 (sm_90) to avoid excessive output that kills Modal builds
        "TORCH_CUDA_ARCH_LIST='8.0;9.0' pip install causal-conv1d --no-build-isolation 2>&1 | grep -v 'ptxas info' | tail -20",
        "TORCH_CUDA_ARCH_LIST='8.0;9.0' pip install mamba-ssm --no-build-isolation 2>&1 | grep -v 'ptxas info' | tail -20",
    )
)

vol = modal.Volume.from_name("peft-paper-results", create_if_missing=True)

# ============================================================
# Model configs
# ============================================================

MODEL_CONFIGS = {
    "qwen": {
        "model_id": "Qwen/Qwen3.5-0.8B",
        "attn_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "ssm_targets": ["in_proj", "out_proj", "gate_proj"],
        "strategies": ["lora_attention", "lora_ssm", "lora_both", "all_linear"],
        "ssm_label": "deltanet",
        "trust_remote_code": True,
        "extra_kwargs": {},
    },
    "jamba": {
        "model_id": "ai21labs/AI21-Jamba-Reasoning-3B",
        "attn_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "ssm_targets": ["in_proj", "out_proj"],
        "strategies": ["lora_attention", "lora_ssm", "lora_both", "all_linear"],
        "ssm_label": "mamba",
        "trust_remote_code": False,
        "extra_kwargs": {"use_mamba_kernels": False},
    },
    "nemotron": {
        "model_id": "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16",
        "attn_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "ssm_targets": ["in_proj", "out_proj"],
        "strategies": ["lora_attention", "lora_ssm", "all_linear"],
        "ssm_label": "mamba2",
        "trust_remote_code": True,
        "extra_kwargs": {},
    },
}

# ============================================================
# Dataset configs
# ============================================================

DATASET_CONFIGS = {
    "code": {
        "name": "Nan-Do/code-search-net-python",
        "config": None,
        "split": "train",
        "min_words": 30,
    },
    "science": {
        "name": "ccdv/pubmed-summarization",
        "config": "document",
        "split": "train",
        "min_words": 50,
    },
    "math": {
        "name": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "min_words": 30,
    },
    "conversation": {
        "name": "HuggingFaceH4/ultrachat_200k",
        "config": None,
        "split": "train_sft",
        "min_words": 50,
    },
}

# ============================================================
# Hyperparams
# ============================================================

NUM_TRAIN = 2000
NUM_TEST = 400
NUM_EPOCHS = 3
LR = 2e-4
LORA_RANK = 16
LORA_ALPHA = 32
MAX_LEN = 256
BATCH_SIZE = 16
SEEDS = [42, 123, 7]
DOMAIN_ORDER = ["code", "science", "conversation", "math"]


# ============================================================
# Main experiment function (runs on H100)
# ============================================================

def _run_experiment_impl(
    model_name: str,
    seeds: list[int],
    run_id: str,
    skip_mmlu: bool = False,
    smoke: bool = False,
) -> dict:
    """Core experiment logic — called by both image-specific Modal functions."""
    import gc
    import math
    import random
    import subprocess
    import sys

    import numpy as np
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_cfg = MODEL_CONFIGS[model_name]
    device = "cuda"

    # Smoke test overrides
    num_train = 50 if smoke else NUM_TRAIN
    num_test = 20 if smoke else NUM_TEST
    max_len = 128 if smoke else MAX_LEN
    num_epochs = 1 if smoke else NUM_EPOCHS
    # Per-model batch size (Nemotron 4B OOMs at 16 due to Mamba2 state expansion)
    model_batch = {"qwen": 16, "jamba": 8, "nemotron": 4}
    batch_size = 4 if smoke else model_batch.get(model_name, 8)

    def log(msg):
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"[{ts}] {msg}", flush=True)

    def clear_mem():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Text extraction per domain ──

    def extract_text(item, domain):
        if domain == "code":
            return item["code"]
        elif domain == "science":
            return item["abstract"]
        elif domain == "math":
            return "Question: " + item["question"] + "\nAnswer: " + item["answer"]
        elif domain == "conversation":
            return " ".join(m["content"] for m in item["messages"])
        return str(item)

    # ── Data loading ──

    def load_datasets(tokenizer, seed=42):
        rng = random.Random(seed)
        need = num_train + num_test + 200  # extra buffer for filtering
        tasks = {}
        for domain in DOMAIN_ORDER:
            cfg = DATASET_CONFIGS[domain]
            log(f"  Loading {domain}: {cfg['name']} (non-streaming, {need} rows)...")
            # Download a fixed slice instead of streaming to avoid httpx client errors
            split_str = f"{cfg['split']}[:{need * 2}]"
            try:
                if cfg["config"]:
                    ds = load_dataset(cfg["name"], cfg["config"], split=split_str)
                else:
                    ds = load_dataset(cfg["name"], split=split_str)
            except Exception as e:
                log(f"    WARN: slice load failed ({e}), trying streaming fallback...")
                kwargs = {"split": cfg["split"], "streaming": True}
                if cfg["config"]:
                    ds_stream = load_dataset(cfg["name"], cfg["config"], **kwargs)
                else:
                    ds_stream = load_dataset(cfg["name"], **kwargs)
                ds = list(ds_stream.take(need * 2))

            texts = []
            for item in ds:
                if len(texts) >= need:
                    break
                text = extract_text(item, domain)
                if len(text.split()) < cfg["min_words"]:
                    continue
                toks = tokenizer(text, max_length=max_len, truncation=True)
                if len(toks["input_ids"]) < 20:
                    continue
                texts.append(text)

            rng.shuffle(texts)
            tasks[domain] = {
                "train": texts[:num_train],
                "test": texts[num_train:num_train + num_test],
            }
            log(f"    {domain}: {len(tasks[domain]['train'])} train, {len(tasks[domain]['test'])} test")
        return tasks

    # ── Target modules ──

    def get_target_modules(strategy):
        if strategy == "lora_attention":
            return model_cfg["attn_targets"]
        elif strategy == "lora_ssm":
            return model_cfg["ssm_targets"]
        elif strategy == "lora_both":
            return model_cfg["attn_targets"] + model_cfg["ssm_targets"]
        elif strategy == "all_linear":
            return "all-linear"
        return "all-linear"

    # ── Training (batched) ──

    def train_epoch(model, tokenizer, texts, optimizer):
        model.train()
        indices = list(range(len(texts)))
        random.shuffle(indices)
        total_loss, n_batches = 0.0, 0

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i + batch_size]
            batch_texts = [texts[j] for j in batch_idx]
            inputs = tokenizer(
                batch_texts, return_tensors="pt", max_length=max_len,
                truncation=True, padding=True,
            ).to(device)

            labels = inputs["input_ids"].clone()
            labels[labels == tokenizer.pad_token_id] = -100

            outputs = model(**inputs, labels=labels)
            outputs.loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += outputs.loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    # ── Evaluation (unbatched for accurate PPL) ──

    @torch.no_grad()
    def eval_ppl(model, tokenizer, texts, n_samples=50):
        model.eval()
        total_loss, total_tokens = 0.0, 0
        for text in texts[:n_samples]:
            inputs = tokenizer(
                text, return_tensors="pt", max_length=max_len, truncation=True,
            ).to(device)
            n_tok = inputs["input_ids"].shape[1]
            if n_tok < 3:
                continue
            outputs = model(**inputs, labels=inputs["input_ids"])
            total_loss += outputs.loss.item() * (n_tok - 1)
            total_tokens += n_tok - 1

        if total_tokens == 0:
            return float("inf")
        return math.exp(min(total_loss / total_tokens, 20))

    # ── MMLU ──

    def run_mmlu(adapter_path, label):
        log(f"  MMLU [{label}]...")
        mmlu_dir = f"/results/{run_id}/mmlu"
        os.makedirs(mmlu_dir, exist_ok=True)

        model_args = f"pretrained={model_cfg['model_id']},dtype=bfloat16"
        model_args += f",trust_remote_code={model_cfg['trust_remote_code']}"
        if adapter_path:
            model_args += f",peft={adapter_path}"

        cmd = [
            sys.executable, "-m", "lm_eval",
            "--model", "hf",
            "--model_args", model_args,
            "--tasks", "mmlu",
            "--num_fewshot", "5",
            "--batch_size", "auto:4",  # limit max batch to avoid OOM on larger models
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
            log(f"    MMLU FAILED: {proc.stderr[-300:]}")

        clear_mem()
        log(f"    Done [{label}] in {elapsed:.1f}s | {result.get('raw_line', 'FAILED')[:80]}")
        return result

    # ── Single CL run (one strategy, one seed) ──

    def run_cl(strategy, tasks, tokenizer, seed):
        log(f"\n{'='*60}")
        log(f"CL: {strategy} | seed={seed} | model={model_name}")
        log(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        load_kwargs = {
            "dtype": torch.bfloat16,
            "trust_remote_code": model_cfg["trust_remote_code"],
        }
        load_kwargs.update(model_cfg["extra_kwargs"])

        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["model_id"], **load_kwargs
        ).to(device)

        target_modules = get_target_modules(strategy)
        log(f"  Targets: {target_modules}")

        lora_config = LoraConfig(
            r=LORA_RANK, lora_alpha=LORA_ALPHA,
            target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM, lora_dropout=0.0,
        )
        model = get_peft_model(model, lora_config)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log(f"  Params: {trainable:,} trainable / {total:,} total ({trainable/total*100:.2f}%)")

        # Baseline PPL
        ppl_matrix = []
        baseline = {}
        for d in DOMAIN_ORDER:
            baseline[d] = eval_ppl(model, tokenizer, tasks[d]["test"])
        ppl_matrix.append(dict(baseline))
        log(f"  Baseline PPL: { {k: f'{v:.1f}' for k,v in baseline.items()} }")

        # Sequential CL training
        train_log = []
        for d in DOMAIN_ORDER:
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()), lr=LR
            )
            t0 = time.perf_counter()
            losses = []
            for epoch in range(num_epochs):
                loss = train_epoch(model, tokenizer, tasks[d]["train"], optimizer)
                losses.append(loss)
            elapsed = time.perf_counter() - t0
            train_log.append({"task": d, "losses": losses, "time": elapsed})
            log(f"  {d}: {losses[0]:.3f} -> {losses[-1]:.3f} ({elapsed:.1f}s)")

            # Eval all domains after each task
            step_ppl = {}
            for ed in DOMAIN_ORDER:
                step_ppl[ed] = eval_ppl(model, tokenizer, tasks[ed]["test"])
            ppl_matrix.append(step_ppl)

        # Save adapter
        adapter_dir = f"/results/{run_id}/adapters/{strategy}_seed{seed}"
        model.save_pretrained(adapter_dir)

        # Forgetting metrics
        forgetting = {}
        for di, d in enumerate(DOMAIN_ORDER):
            ppls = [ppl_matrix[s][d] for s in range(len(ppl_matrix))]
            best = min(ppls[1:])
            final = ppls[-1]
            forgetting[d] = {
                "best": float(best), "final": float(final),
                "delta": float(final - best),
            }

        result = {
            "strategy": strategy,
            "model": model_cfg["model_id"],
            "ssm_type": model_cfg["ssm_label"],
            "seed": seed,
            "lora_rank": LORA_RANK,
            "num_train": num_train,
            "num_test": num_test,
            "max_len": max_len,
            "batch_size": batch_size,
            "trainable_params": trainable,
            "total_params": total,
            "param_pct": f"{trainable/total*100:.2f}%",
            "baseline_ppl": {k: float(v) for k, v in baseline.items()},
            "ppl_matrix": [
                {k: float(v) for k, v in step.items()} for step in ppl_matrix
            ],
            "forgetting": forgetting,
            "avg_forgetting": float(
                np.mean([v["delta"] for v in forgetting.values()])
            ),
            "avg_final_ppl": float(
                np.mean([ppl_matrix[-1][d] for d in DOMAIN_ORDER])
            ),
            "train_log": train_log,
            "adapter_path": adapter_dir,
        }

        del model, optimizer
        clear_mem()
        return result

    # ════════════════════════════════════════════════════════════
    # MAIN EXPERIMENT LOGIC
    # ════════════════════════════════════════════════════════════

    t_start = time.perf_counter()

    log("=" * 70)
    log(f"PEFT PAPER EXPERIMENT: {model_cfg['model_id']}")
    log(f"Run ID: {run_id}")
    log(f"GPU: {torch.cuda.get_device_name()}")
    log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    log(f"Strategies: {model_cfg['strategies']}")
    log(f"Seeds: {seeds}")
    log(f"Train: {num_train}/domain | Test: {num_test}/domain | MaxLen: {max_len}")
    log(f"LoRA: r={LORA_RANK}, alpha={LORA_ALPHA} | Batch: {batch_size} | Epochs: {num_epochs}")
    log(f"Smoke: {smoke}")
    log("=" * 70)

    # Load tokenizer and data (once, shared across all runs)
    log("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg["model_id"], trust_remote_code=model_cfg["trust_remote_code"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("Loading datasets...")
    tasks = load_datasets(tokenizer, seed=42)

    # Run CL experiments (with resume support)
    os.makedirs(f"/results/{run_id}", exist_ok=True)
    all_cl = []

    # Resume: load existing results if available
    cl_path = f"/results/{run_id}/cl_results.json"
    if os.path.exists(cl_path):
        with open(cl_path) as f:
            all_cl = json.load(f)
        done_keys = {(r["strategy"], r["seed"]) for r in all_cl}
        log(f"Resuming: {len(all_cl)} runs already completed")
    else:
        done_keys = set()

    for strategy in model_cfg["strategies"]:
        for seed in seeds:
            if (strategy, seed) in done_keys:
                log(f"Skipping {strategy} seed={seed} (already done)")
                continue
            result = run_cl(strategy, tasks, tokenizer, seed)
            all_cl.append(result)
            # Incremental save
            with open(f"/results/{run_id}/cl_results.json", "w") as f:
                json.dump(all_cl, f, indent=2, default=str)
            vol.commit()

    # CL Summary
    log(f"\n{'='*70}")
    log("CL SUMMARY")
    log(f"{'Strategy':<22} {'Seeds':>5} {'Avg PPL':>14} {'Avg Forget':>14}")
    log("-" * 58)

    from collections import defaultdict
    by_strategy = defaultdict(list)
    for r in all_cl:
        by_strategy[r["strategy"]].append(r)

    summary = {}
    for s, runs in by_strategy.items():
        ppls = [r["avg_final_ppl"] for r in runs]
        forgets = [r["avg_forgetting"] for r in runs]
        pm, ps = float(np.mean(ppls)), float(np.std(ppls))
        fm, fs = float(np.mean(forgets)), float(np.std(forgets))
        log(f"  {s:<20} {len(runs):>5}  {pm:>8.1f}+/-{ps:<5.1f}  {fm:>8.1f}+/-{fs:<5.1f}")
        summary[s] = {
            "n_seeds": len(runs),
            "avg_final_ppl_mean": pm,
            "avg_final_ppl_std": ps,
            "avg_forgetting_mean": fm,
            "avg_forgetting_std": fs,
            "param_pct": runs[0]["param_pct"],
        }

    # MMLU (seed=42 adapters only)
    mmlu_results = {}
    if not skip_mmlu:
        log(f"\n{'='*70}")
        log("MMLU EVALUATION (seed=42 adapters)")
        log(f"{'='*70}")

        mmlu_results["base"] = run_mmlu(None, "base")
        for r in all_cl:
            if r["seed"] == seeds[0]:  # first seed (usually 42)
                label = f"{r['strategy']}_seed{r['seed']}"
                mmlu_results[label] = run_mmlu(r["adapter_path"], label)

    # Final output
    total_time = time.perf_counter() - t_start

    output = {
        "run_id": run_id,
        "model": model_name,
        "model_id": model_cfg["model_id"],
        "ssm_type": model_cfg["ssm_label"],
        "gpu": torch.cuda.get_device_name(),
        "total_time_seconds": total_time,
        "total_time_human": f"{total_time/60:.1f} minutes",
        "config": {
            "num_train": num_train,
            "num_test": num_test,
            "max_len": max_len,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lr": LR,
            "seeds": seeds,
        },
        "cl_results": all_cl,
        "cl_summary": summary,
        "mmlu_results": mmlu_results,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    with open(f"/results/{run_id}/results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    vol.commit()

    log(f"\n{'='*70}")
    log(f"DONE in {total_time/60:.1f} minutes")
    log(f"Results saved to volume: /results/{run_id}/results.json")
    log(f"{'='*70}")

    return output


@app.function(image=image, gpu="H100", timeout=25200, volumes={"/results": vol})
def run_model_experiment(model_name: str, seeds: list[int], run_id: str,
                         skip_mmlu: bool = False, smoke: bool = False) -> dict:
    return _run_experiment_impl(model_name, seeds, run_id, skip_mmlu, smoke)


@app.function(image=nemotron_image, gpu="H100", timeout=25200, volumes={"/results": vol})
def run_nemotron_experiment(seeds: list[int], run_id: str,
                            skip_mmlu: bool = False, smoke: bool = False) -> dict:
    return _run_experiment_impl("nemotron", seeds, run_id, skip_mmlu, smoke)


# ============================================================
# Local entrypoint
# ============================================================

@app.local_entrypoint()
def main(
    model: str = "qwen",
    smoke: bool = False,
    skip_mmlu: bool = False,
    seeds: str = "42,123,7",
    run_id: str = "",
):
    if model not in MODEL_CONFIGS:
        print(f"Unknown model: {model}. Choose: {list(MODEL_CONFIGS.keys())}")
        return

    seed_list = [int(s.strip()) for s in seeds.split(",")]

    if not run_id:
        prefix = "smoke" if smoke else "paper"
        run_id = f"{prefix}_{model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    cfg = MODEL_CONFIGS[model]
    n_strategies = len(cfg["strategies"])
    n_runs = n_strategies * len(seed_list)

    # Time estimates (minutes per CL run on H100)
    time_per_run = {"qwen": 2.5, "nemotron": 7, "jamba": 7.5}
    if smoke:
        est_min = n_runs * 0.5
    else:
        est_min = n_runs * time_per_run.get(model, 5)
    if not skip_mmlu:
        est_min += (n_strategies + 1) * 5  # ~5 min per MMLU eval

    cost_per_hr = 3.95  # H100
    est_cost = (est_min / 60) * cost_per_hr

    print(f"{'='*50}")
    print(f"PEFT Paper Experiment")
    print(f"{'='*50}")
    print(f"Run ID:     {run_id}")
    print(f"Model:      {cfg['model_id']}")
    print(f"Strategies: {cfg['strategies']}")
    print(f"Seeds:      {seed_list}")
    print(f"Total runs: {n_runs} CL + {'MMLU' if not skip_mmlu else 'no MMLU'}")
    print(f"Est. time:  ~{est_min:.0f} min")
    print(f"Est. cost:  ~${est_cost:.2f}")
    print(f"Smoke:      {smoke}")
    print()

    if model == "nemotron":
        result = run_nemotron_experiment.remote(
            seeds=seed_list,
            run_id=run_id,
            skip_mmlu=skip_mmlu,
            smoke=smoke,
        )
    else:
        result = run_model_experiment.remote(
            model_name=model,
            seeds=seed_list,
            run_id=run_id,
            skip_mmlu=skip_mmlu,
            smoke=smoke,
        )

    # Save locally
    local_dir = Path("results/modal")
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / f"{run_id}.json"
    with open(local_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\nLocal results:  {local_path}")
    print(f"Volume results: modal volume get peft-paper-results {run_id}/results.json")
    print(f"Adapters:       modal volume ls peft-paper-results {run_id}/adapters/")
