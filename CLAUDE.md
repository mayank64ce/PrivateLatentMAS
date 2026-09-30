# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Research code for **LatentMAS** (ICML 2026, arXiv 2511.20639): a training-free multi-agent reasoning framework where agents communicate through the model's latent space (KV cache / hidden states) instead of text. The repo compares three methods — single-agent `baseline`, token-space `text_mas`, and latent-space `latent_mas` — on GSM8K, AIME24/25, GPQA-Diamond, ARC-Easy/Challenge, MBPP+, HumanEval+, and MedQA.

There is no build system, test suite, or linter. Everything runs through `run.py`.

## Setup

```bash
conda create -n latentmas python=3.10 -y && conda activate latentmas
pip install -r requirements.txt
pip install vllm          # effectively required, see gotchas below
export HF_HOME=/path/to/huggingface   # models/datasets download here
```

## Running experiments

```bash
# Baseline / TextMAS / LatentMAS (HF backend; use this to reproduce published numbers)
python run.py --method baseline   --model_name Qwen/Qwen3-14B --task gsm8k --max_samples -1 --max_new_tokens 2048
python run.py --method text_mas   --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --max_samples -1 --max_new_tokens 2048
python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential --latent_steps 20 --max_new_tokens 2048

# Quick smoke test: small --max_samples and --generate_bs
python run.py --method latent_mas --model_name Qwen/Qwen3-4B --task gsm8k --max_samples 4 --generate_bs 2 --latent_steps 10

# LatentMAS hybrid vLLM pipeline (vLLM on --device, auxiliary HF model on --device2)
CUDA_VISIBLE_DEVICES=0,1 python run.py --method latent_mas --model_name Qwen/Qwen3-14B --task gsm8k --prompt sequential \
  --max_samples -1 --max_new_tokens 2048 --use_vllm --use_second_HF_model --enable_prefix_caching --device2 cuda:1
```

Key flags: `--prompt {sequential,hierarchical}` selects the MAS topology; `--latent_steps` (0–80, default 0 — tune per task) is the number of latent thoughts per non-judger agent; `--latent_space_realign` is a per-task/model hyperparameter; `--think` appends `<think>` to prompts; `--generate_bs` is the batch size. Output is a per-problem trace to stdout followed by a final JSON line with accuracy and timing.

## Architecture

**Pipeline** (`run.py`): parse args → `ModelWrapper` → method object → iterate a `data.py` loader in batches of `generate_bs` → `method.run_batch(items)` (or `run_batch_vllm` for `latent_mas` + `--use_vllm`) → each result dict has `question/gold/prediction/correct/agents` → accuracy = mean of `correct`.

**Agents** (`methods/__init__.py`): every MAS method uses the fixed chain `default_agents()` = Planner → Critic → Refiner → Judger. Only the Judger produces the final decoded answer in `latent_mas`.

**Prompts** (`prompts.py`): one builder per (method × topology) — `build_agent_message_{sequential,hierarchical}_latent_mas`, `build_agent_messages_{sequential,hierarchical}_text_mas`, `build_agent_messages_single_agent`. Each branches on `role` and on `args.task` (task groups: math `gsm8k/aime*`, multiple-choice `arc_*/gpqa/medqa`, code `mbppplus/humanevalplus`, plus an unused `winogrande`). Adding a task means touching every builder here, plus `data.py`, the `--task` choices and loader dispatch in `run.py`, and answer checking in each method. `get_system_message(args)` returns the original Qwen system prompt for Qwen models and a neutral one otherwise (Llama is supported; see the README's Llama section).

**Latent communication** (`models.py` + `methods/latent_mas.py`):
- HF path: for each non-judger agent, `ModelWrapper.generate_latent_batch` runs a forward pass on the agent prompt, then for `latent_steps` iterations feeds the last-layer hidden state back as an input embedding (after `_apply_latent_realignment`), extending a shared `past_key_values`. The accumulated KV cache is passed to the Judger's `generate_text_batch`, so later agents "see" earlier agents only through the KV cache (prompts are built with `context=""`).
- Realignment: `_build_latent_realign_matrix` solves a ridge least-squares map from output-embedding space (`lm_head`) to input-embedding space; hidden states are mapped through it and rescaled to the mean input-embedding norm. Without `--latent_space_realign` the matrix is identity but the norm rescaling still applies.
- vLLM path (`run_batch_vllm`): a second HF model on `--device2` runs the latent rollout via `generate_latent_batch_hidden_state`, which returns the input embeddings it fed in. These are concatenated and spliced into the Judger's prompt embeddings right before the user message content (located by searching the rendered prompt for the content, so it works for any chat template), then passed to vLLM with `prompt_embeds` (`enable_prompt_embeds=True`). KV cache is not transferred to vLLM, only embeddings.
- `latent_only` / `sequential_info_only` (ablations that truncate the KV cache to only the latest agent's contribution) are read via `getattr(args, ...)` but are not exposed as CLI flags.

**Evaluation**: answers are extracted with `utils.extract_gsm8k_answer` (last `\boxed{}` or last number; also used for MC letters) and compared after `normalize_answer`. AIME compares as ints. Code tasks extract the last ```python block, append the test code from `gold`, and `exec` it in a subprocess with a 10s timeout (`utils.run_with_timeout`).

## Gotchas

- `methods/latent_mas.py` imports `vllm` unconditionally, so `run.py` fails without vLLM installed even for HF-only runs. `models.py` imports `matplotlib`, which is not in `requirements.txt`.
- `--model_name` is restricted by `choices` in `run.py`; add new models there.
- The env has `transformers` 5.x: KV caches are `DynamicCache` objects (not tuples), and `generate()` rejects `cache_position`. `models.py` handles both (`_past_length`, `_TRANSFORMERS_MAJOR`).
- `run_batch_vllm` uses only the generic answer check, so code tasks and AIME are not scored correctly on the LatentMAS vLLM path.
- `load_medqa` reads `./data/medqa.json` relative to CWD; run from the repo root. Other datasets come from the HF Hub.
- vLLM results differ slightly from HF; the README says to use the HF backend for official numbers.
- `example_logs/` has reference LatentMAS traces showing the expected stdout format.
