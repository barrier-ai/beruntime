# beruntime

**Barrier Research Runtime** (`beruntime`) provides a Python `ctypes` interface to the compiled `brr_*` native C inference engine for `Klorn-Tinystories-31M`.

- **Zero PyTorch/NumPy inference dependency**: Requires only Python `>=3.10` and `tokenizers>=0.15.0`.
- **Prebuilt native binaries bundled**: Ships compiled AVX2/FMA binaries (`brr_runtime.dll` for Windows x86_64 and `libbrr_runtime.so` for Linux x86_64).

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/barrier-ai/beruntime.git
```

## Quickstart

Download `model.safetensors` and `tokenizer.json` from [`brr-ai/Klorn-Tinystories-31M`](https://huggingface.co/brr-ai/Klorn-Tinystories-31M):

```bash
pip install huggingface_hub
hf download brr-ai/Klorn-Tinystories-31M --local-dir ./Klorn-Tinystories-31M
```

Then run inference in Python:

```python
from beruntime import KlornModel

with KlornModel("./Klorn-Tinystories-31M") as model:
    res = model.generate(
        "Once upon a time, a little",
        max_new_tokens=256,
        temperature=0.0,
    )
    print(f"{res.decode_tok_per_sec:.2f} tok/s | {res.prompt + res.text}")
```

## License

Non-commercial research and evaluation only under [Barrier Research Evaluation License (`barrier-eval-1.0`)](LICENSE).
