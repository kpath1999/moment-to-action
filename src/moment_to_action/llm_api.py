import argparse
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set transformers verbosity for better debugging
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# CONFIGURATION
@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.1


@dataclass
class AppConfig:
    model_id: str
    model_dir: str
    device: str = field(default_factory=_default_device)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    backend: str = "huggingface"  # match key in LLM registry
    hf_token: str | None = None


class LLMBase(ABC):
    """Minimal LLM interface with a tiny backend registry."""

    _registry: ClassVar[dict[str, type["LLMBase"]]] = {}

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    @classmethod
    def register(cls, name: str) -> Any:
        def decorator(subclass: type["LLMBase"]) -> type["LLMBase"]:
            cls._registry[name] = subclass
            return subclass

        return decorator

    @classmethod
    def create(cls, config: AppConfig) -> "LLMBase":
        backend = config.backend.lower()
        if backend not in cls._registry:
            available = ", ".join(sorted(cls._registry))
            msg = f"Unknown backend '{backend}'. Available: {available}"
            raise ValueError(msg)
        return cls._registry[backend](config)

    @abstractmethod
    def load(self) -> None:
        """Load model resources."""

    @abstractmethod
    def run(self, system_prompt: str, user_prompt: str) -> str:
        """Generate one assistant response."""


@LLMBase.register("huggingface")
class HuggingFaceLLM(LLMBase):
    """Small Hugging Face text-generation backend."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self.tokenizer = None
        self.model = None
        self.torch_device = torch.device(config.device)

    def load(self) -> None:
        token = (
            self.config.hf_token
            or os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_id,
            cache_dir=self.config.model_dir,
            token=token,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_id,
            cache_dir=self.config.model_dir,
            token=token,
        )
        self.model.to(self.torch_device)
        self.model.eval()

    def run(self, system_prompt: str, user_prompt: str) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not loaded. Call load() first.")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Prefer model chat template when available, fall back to plain text prompt.
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"System: {system_prompt}\n" f"User: {user_prompt}\n" "Assistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.torch_device)

        # Build generation kwargs - only include supported parameters
        gen_kwargs = {
            "max_new_tokens": self.config.generation.max_new_tokens,
            "do_sample": self.config.generation.do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Add sampling parameters only if do_sample=True
        if self.config.generation.do_sample:
            gen_kwargs.update(
                {
                    "temperature": self.config.generation.temperature,
                    "top_p": self.config.generation.top_p,
                    "top_k": self.config.generation.top_k,
                    "repetition_penalty": self.config.generation.repetition_penalty,
                }
            )

        outputs = self.model.generate(**inputs, **gen_kwargs)

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple LLM runner")
    parser.add_argument(
        "--backend", default="huggingface", help="LLM backend registry key"
    )
    parser.add_argument(
        "--model-id", default="gpt2", help="Model id from Hugging Face Hub"
    )
    parser.add_argument(
        "--model-dir", default="./models", help="Local model cache directory"
    )
    parser.add_argument(
        "--device", default=_default_device(), choices=["cpu", "cuda", "mps"]
    )
    parser.add_argument(
        "--hf-token", default=None, help="HF token (or use HF_TOKEN env var)"
    )
    parser.add_argument(
        "--system", default="You are a helpful assistant.", help="System prompt"
    )
    parser.add_argument("--prompt", required=True, help="User prompt")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose transformers logging"
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> AppConfig:
    return AppConfig(
        model_id=args.model_id,
        model_dir=args.model_dir,
        device=args.device,
        generation=GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
        ),
        backend=args.backend,
        hf_token=args.hf_token,
    )


"""
ungated models:

gpt2
gpt2-medium
gpt2-large
distilgpt2
Qwen/Qwen2.5-1.5B-Instruct
Qwen/Qwen2.5-3B-Instruct
google/gemma-2b-it
microsoft/phi-2
stabilityai/stablelm-2-1_6b-chat
mistralai/Mistral-7B-Instruct-v0.3
tiiuae/falcon-7b-instruct
"""

# USAGE
if __name__ == "__main__":
    args = parse_args()

    # Set verbosity if requested
    if args.verbose:
        os.environ["TRANSFORMERS_VERBOSITY"] = "info"

    cfg = build_config(args)
    llm = LLMBase.create(cfg)
    llm.load()
    print(llm.run(args.system, args.prompt))
