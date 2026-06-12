# Modal vLLM Inference — OpenAI-compatible LLM server

Deploys [Google Gemma 4 26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it) on a single
H200 GPU via [Modal](https://modal.com) and exposes an OpenAI-compatible REST API powered by
[vLLM](https://docs.vllm.ai).

Based on the official [Modal vLLM inference example](https://modal.com/docs/examples/vllm_inference).

## Requirements

- Python 3.11+
- A [Modal account](https://modal.com/signup) with GPU quota (H200)
- A [Hugging Face account](https://huggingface.co) with access to the Gemma 4 gated repos

## Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Authenticate with Modal
modal setup

# 3. Store your Hugging Face token as a Modal named secret
modal secret create huggingface HF_TOKEN=<your-huggingface-token>
```

> The Gemma 4 model is gated. Accept the licence at
> https://huggingface.co/google/gemma-4-26B-A4B-it before deploying.

## Deploy

```bash
modal deploy vllm_inference.py
```

After a successful deploy you will see a URL like:

```
https://<workspace>--vibe2blog-backend-serve.modal.run
```

The first deploy builds the container image and downloads the model weights (~52 GB).
Subsequent deploys reuse the cached image and weights.

## Register API Clients

The public vLLM routes require an `X-API-Key` header. Keys are stored as SHA-256
hashes in a SQLite registry and should never be committed.

For local registry workflows:

```bash
python -m register_apps --name="vibe2blog" --expired=2027-01-01
```

For the deployed Modal backend, register directly into the persistent
`app-registry` volume:

```bash
modal run vllm_inference.py::register_client --name vibe2blog --expired 2027-01-01
```

The command prints the raw API key once. Store it in the Vibe2Blog Space secret
as `MODAL_VLLM_API_KEY`.

## Smoke-test (modal run)

```bash
modal run vllm_inference.py
```

This spins up a fresh replica, runs a health check, then sends a blog-polishing
completion request and streams the output to your terminal. No GPU is required
on your local machine.

## Health check

```bash
curl https://<workspace>--vibe2blog-backend-serve.modal.run/healthz
# → {"status": "ok"}
```

> `/healthz` blocks until vLLM is ready and returns JSON. `/health` is intercepted by Modal's edge and returns `200` with an empty body.

## Chat completions (non-streaming)

```bash
curl -s https://<workspace>--vibe2blog-backend-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <api-key>" \
  -d '{
    "model": "llm",
    "stream": false,
    "messages": [
      {
        "role": "system",
        "content": "You are a concise assistant."
      },
      {
        "role": "user",
        "content": "What is vLLM?"
      }
    ]
  }' | python3 -m json.tool
```

## Vibe2Blog integration

Set these two environment variables in your Vibe2Blog deployment:

```bash
MODAL_VLLM_BASE_URL=https://<workspace>--vibe2blog-backend-serve.modal.run/v1
MODAL_VLLM_MODEL=llm
MODAL_VLLM_API_KEY=<api-key-from-register-client>
```

The server registers the model under both its full HuggingFace name
(`google/gemma-4-26B-A4B-it`) and the short alias `llm`, so
`MODAL_VLLM_MODEL=llm` always works regardless of which model is deployed.

## Client

```bash
# single prompt (streaming)
python openai_compatible/client.py

# interactive chat
python openai_compatible/client.py --chat

# custom prompt, no streaming
python openai_compatible/client.py \
  --prompt "Explain gradient descent in one paragraph." \
  --no-stream
```

Full CLI reference:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | auto-detect | Override model ID |
| `--app-name` | `vibe2blog-backend` | Modal app name |
| `--function-name` | `serve` | Modal function name |
| `--workspace` | current profile | Modal workspace |
| `--environment` | current env | Modal environment |
| `--prompt` | limerick prompt | User message |
| `--system-prompt` | poetic assistant | System message |
| `--temperature` | 0.7 | Sampling temperature |
| `--max-tokens` | unlimited | Max tokens to generate |
| `--no-stream` | — | Disable SSE streaming |
| `--chat` | — | Interactive chat loop |

## Configuration

Edit the top of `vllm_inference.py` to change:

| Variable | Default | Effect |
|----------|---------|--------|
| `MODEL_NAME` | `google/gemma-4-26B-A4B-it` | HuggingFace model repo |
| `MODEL_REVISION` | pinned SHA | Exact weights revision |
| `FAST_BOOT` | `False` | `True` → skip JIT compilation for faster cold starts |
| `N_GPU` | `1` | Number of H200s (increase for larger models) |
| `VLLM_API_KEY_AUTH_ENABLED` | `true` | Require `X-API-Key` for `/v1/*` routes |
| `APP_REGISTRY_DB` | `/data/app_registry.sqlite3` | SQLite registry path in Modal |

## Run tests

```bash
pytest
```

Tests run locally without any GPU or Modal credentials. They cover:
- Configuration constants (model name/revision pinning, port, GPU count)
- SSE stream parsing in `_send_request` (content, reasoning, [DONE] sentinel, bad object type)
- SQLite app registry registration, key rotation, and expiry checks

## Project structure

```
.
├── vllm_inference.py          # Modal app: container image, serve function, local_entrypoint
├── openai_compatible/
│   └── client.py              # CLI client using the openai Python SDK
├── tests/
│   └── test_vllm_inference.py # Unit tests (no GPU required)
├── docs/
│   └── api-docs.md  # Full API reference (endpoints, request/response schemas)
├── requirements.txt           # Pinned local dependencies
└── pytest.ini                 # pytest configuration
```
