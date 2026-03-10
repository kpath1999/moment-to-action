# LLM API Usage Guide

## Overview

`llm_api.py` is a minimal command-line interface for running Hugging Face language models locally. It provides:

- **Easy model swapping** via `--model-id` parameter
- **Device support** for CPU, CUDA (NVIDIA), and MPS (Apple Silicon)
- **Backend registry** pattern for adding new model providers
- **Chat template support** with fallback to plain text prompts
- **Token authentication** for gated models

## Prerequisites

### 1. Python Environment
```bash
# Create conda environment
conda create -n m2a python=3.10
conda activate m2a
```

### 2. Install Dependencies
```bash
# Install PyTorch (ensure version ≥ 2.4 for latest transformers)
pip install torch torchvision torchaudio

# Install Transformers
pip install transformers huggingface_hub
```

### 3. Optional: Hugging Face Token
For gated models (like Llama), set your HF token:
```bash
export HF_TOKEN="hf_your_token_here"
# Or pass via --hf-token flag
```

## Basic Usage

### Quick Test
```bash
python llm_api.py --prompt "What is artificial intelligence?"
```

### With Custom Model
```bash
python llm_api.py \
  --model-id Qwen/Qwen2.5-1.5B-Instruct \
  --device mps \
  --prompt "Explain machine learning in one sentence" \
  --do-sample
```

## CLI Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--backend` | `huggingface` | LLM backend (extensible) |
| `--model-id` | `gpt2` | Model from Hugging Face Hub |
| `--model-dir` | `./models` | Local cache directory |
| `--device` | auto-detect | Device: `cpu`, `cuda`, or `mps` |
| `--hf-token` | `None` | HF token (or use `HF_TOKEN` env var) |
| `--system` | `"You are a helpful assistant."` | System prompt |
| `--prompt` | **required** | User prompt |
| `--max-new-tokens` | `80` | Max tokens to generate |
| `--temperature` | `0.7` | Sampling temperature |
| `--do-sample` | `False` | Enable sampling (add flag to enable) |

## Recommended Models (No Access Required)

### Small/Fast (good for testing)
- `gpt2` - 124M parameters
- `distilgpt2` - 82M parameters
- `Qwen/Qwen2.5-1.5B-Instruct` - 1.5B, instruction-tuned

### Better Quality
- `microsoft/phi-2` - 2.7B parameters
- `Qwen/Qwen2.5-3B-Instruct` - 3B parameters
- `google/gemma-2b-it` - 2B instruction model

## Examples

### Different Models
```bash
# Fast baseline
python llm_api.py --model-id gpt2 --prompt "Hello world"

# Better instruction following
python llm_api.py --model-id Qwen/Qwen2.5-1.5B-Instruct --prompt "Write a Python function to add two numbers"

# Larger model with sampling
python llm_api.py --model-id microsoft/phi-2 --prompt "Explain quantum computing" --do-sample --max-new-tokens 100
```

### Custom System Prompts
```bash
python llm_api.py \
  --system "You are a concise technical writer." \
  --prompt "Explain Docker containers" \
  --max-new-tokens 50
```

## Troubleshooting

### PyTorch Version Error
```
Error: PyTorch >= 2.4 is required but found 2.2.0
```
**Fix:** `pip install --upgrade torch`

### Gated Model Access
```
Error: Access to model X is restricted
```
**Fix:** Visit the model page on Hugging Face and request access

### Token Authentication
```
Error: 401 Unauthorized
```
**Fix:** Set `HF_TOKEN` environment variable or use `--hf-token`

### MPS Device Not Available
```
Error: MPS not available
```
**Fix:** Use `--device cpu` on older Macs or non-Apple Silicon

## Architecture

The script uses a simple registry pattern:
- `LLMBase` - Abstract base class
- `HuggingFaceLLM` - Concrete implementation for HF models
- Registry allows adding new backends (OpenAI, Anthropic, etc.)

Flow: CLI args → `AppConfig` → `LLMBase.create()` → `load()` → `run()`