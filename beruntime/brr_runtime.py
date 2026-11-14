"""Lightweight ctypes Python binding for the brr_* C inference runtime (beruntime)."""

from __future__ import annotations

import ctypes
import math
import platform
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from tokenizers import Tokenizer


class _BrrLib:
    """Thin typed binding layer around the brr_ C ABI shared library."""

    def __init__(self, lib_path: Path) -> None:
        if not lib_path.exists():
            raise FileNotFoundError(f"Runtime library not found: {lib_path}")
        self._dll = ctypes.CDLL(str(lib_path))

        self.version = self._dll.brr_version
        self.version.argtypes = []
        self.version.restype = ctypes.c_char_p

        self.last_error = self._dll.brr_last_error
        self.last_error.argtypes = []
        self.last_error.restype = ctypes.c_char_p

        self.create = self._dll.brr_create
        self.create.argtypes = [ctypes.c_char_p]
        self.create.restype = ctypes.c_void_p

        self.destroy = self._dll.brr_destroy
        self.destroy.argtypes = [ctypes.c_void_p]
        self.destroy.restype = None

        self.reset = self._dll.brr_reset
        self.reset.argtypes = [ctypes.c_void_p]
        self.reset.restype = None

        self.vocab_size = self._dll.brr_vocab_size
        self.vocab_size.argtypes = [ctypes.c_void_p]
        self.vocab_size.restype = ctypes.c_int32

        self.state_bytes = self._dll.brr_state_bytes
        self.state_bytes.argtypes = [ctypes.c_void_p]
        self.state_bytes.restype = ctypes.c_size_t

        self.forward_token = self._dll.brr_forward_token
        self.forward_token.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.forward_token.restype = ctypes.c_int32

    def error_message(self) -> str:
        raw = self.last_error()
        return raw.decode("utf-8", errors="replace") if raw else "unknown runtime error"


@dataclass(frozen=True)
class GenerationResult:
    """Output and performance telemetry returned by KlornModel.generate()."""

    prompt: str
    text: str
    generated_ids: list[int]
    prompt_tokens: int
    generated_tokens: int
    prefill_seconds: float
    decode_seconds: float
    decode_tok_per_sec: float


def _default_lib_name() -> str:
    sys_name = platform.system()
    if sys_name == "Windows":
        return "brr_runtime.dll"
    if sys_name == "Darwin":
        return "libbrr_runtime.dylib"
    return "libbrr_runtime.so"


def _resolve_lib_path(model_dir: Path, lib_path: str | Path | None) -> Path:
    if lib_path is not None:
        return Path(lib_path)
    lib_name = _default_lib_name()
    local_candidate = model_dir / lib_name
    if local_candidate.exists():
        return local_candidate
    bundled_candidate = Path(__file__).resolve().parent / "_bin" / lib_name
    if bundled_candidate.exists():
        return bundled_candidate
    raise FileNotFoundError(
        f"Runtime binary '{lib_name}' not found in '{model_dir}' or '{bundled_candidate.parent}'"
    )


def _sample_next_token(
    logits: Sequence[float],
    temperature: float,
    top_p: float,
    rng: random.Random,
) -> int:
    if temperature <= 0.0:
        return max(range(len(logits)), key=logits.__getitem__)

    scaled = [v / temperature for v in logits]
    max_logit = max(scaled)
    probs = [math.exp(v - max_logit) for v in scaled]
    total = sum(probs)
    inv_total = 1.0 / total
    norm_probs = [p * inv_total for p in probs]

    if 0.0 < top_p < 1.0:
        order = sorted(range(len(norm_probs)), key=norm_probs.__getitem__, reverse=True)
        kept_indices: list[int] = []
        kept_probs: list[float] = []
        cum = 0.0
        for idx in order:
            kept_indices.append(idx)
            kept_probs.append(norm_probs[idx])
            cum += norm_probs[idx]
            if cum >= top_p:
                break
        return rng.choices(kept_indices, weights=kept_probs, k=1)[0]

    return rng.choices(range(len(norm_probs)), weights=norm_probs, k=1)[0]


class KlornModel:
    """Safe Python interface over the compiled Klorn-Tinystories-31M C runtime."""

    EOS_TOKEN_ID = 2

    def __init__(
        self,
        model_dir: str | Path,
        lib_path: str | Path | None = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        weights_path = self.model_dir / "model.safetensors"
        tok_path = self.model_dir / "tokenizer.json"

        if not weights_path.exists():
            raise FileNotFoundError(f"Safetensors file not found: {weights_path}")
        if not tok_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tok_path}")

        resolved_lib = _resolve_lib_path(self.model_dir, lib_path)
        self._lib = _BrrLib(resolved_lib)
        self._ctx = self._lib.create(str(weights_path).encode("utf-8"))
        if not self._ctx:
            raise RuntimeError(f"brr_create failed: {self._lib.error_message()}")

        self.tokenizer = Tokenizer.from_file(str(tok_path))
        self.vocab_size: int = int(self._lib.vocab_size(self._ctx))
        self.state_bytes: int = int(self._lib.state_bytes(self._ctx))
        self.runtime_version: str = self._lib.version().decode("ascii")
        self._logits_buf = (ctypes.c_float * self.vocab_size)()

    def reset(self) -> None:
        """Reset all recurrent DDA matrices, SWA KV caches, and document boundary state."""
        self._ensure_open()
        self._lib.reset(self._ctx)

    def forward_token(self, token_id: int) -> list[float]:
        """Advance the model by one token ID and return a list of float32 logits."""
        self._ensure_open()
        rc = self._lib.forward_token(self._ctx, int(token_id), self._logits_buf)
        if rc != 0:
            raise RuntimeError(f"brr_forward_token failed: {self._lib.error_message()}")
        return [self._logits_buf[i] for i in range(self.vocab_size)]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 0.9,
        seed: int | None = None,
        reset_before: bool = True,
        stop_on_eos: bool = True,
    ) -> GenerationResult:
        """Tokenize prompt, run prefill + autoregressive decode, and return text + telemetry."""
        self._ensure_open()
        if reset_before:
            self.reset()

        prompt_ids = self.tokenizer.encode(prompt).ids
        if not prompt_ids:
            prompt_ids = [self.EOS_TOKEN_ID]

        t0 = time.perf_counter()
        last_logits: list[float] = []
        for tid in prompt_ids:
            last_logits = self.forward_token(tid)
        t_prefill = time.perf_counter() - t0

        rng = random.Random(seed)
        generated_ids: list[int] = []
        t1 = time.perf_counter()
        for _ in range(max_new_tokens):
            next_id = _sample_next_token(last_logits, temperature=temperature, top_p=top_p, rng=rng)
            generated_ids.append(next_id)
            if stop_on_eos and next_id == self.EOS_TOKEN_ID:
                self.forward_token(next_id)
                break
            last_logits = self.forward_token(next_id)
        t_decode = time.perf_counter() - t1

        tok_per_sec = (len(generated_ids) / t_decode) if t_decode > 0.0 else 0.0
        prefix_text = self.tokenizer.decode(prompt_ids, skip_special_tokens=True)
        full_text = self.tokenizer.decode(prompt_ids + generated_ids, skip_special_tokens=True)
        if full_text.startswith(prefix_text):
            decoded = full_text[len(prefix_text) :]
        else:
            decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return GenerationResult(
            prompt=prompt,
            text=decoded,
            generated_ids=generated_ids,
            prompt_tokens=len(prompt_ids),
            generated_tokens=len(generated_ids),
            prefill_seconds=t_prefill,
            decode_seconds=t_decode,
            decode_tok_per_sec=tok_per_sec,
        )

    def close(self) -> None:
        if getattr(self, "_ctx", None):
            self._lib.destroy(self._ctx)
            self._ctx = None

    def _ensure_open(self) -> None:
        if not self._ctx:
            raise RuntimeError("KlornModel runtime context has already been closed")

    def __enter__(self) -> "KlornModel":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()
