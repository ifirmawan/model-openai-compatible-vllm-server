# --- # pytest: false # --- #
# Run OpenAI-compatible LLM inference with Gemma and vLLM
#
# Deploy:   modal deploy vllm_inference.py
# Test run: modal run vllm_inference.py

import json
import os
from typing import Any

import aiohttp
import modal
from dotenv import load_dotenv

load_dotenv()  # loads .env into os.environ

HF_TOKEN = os.environ["HF_TOKEN"]  # required — set in .env

# ## Container image
# Installed on top of the CUDA 12.9 base image provided by Modal.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",   # faster model transfers from HF
            "VLLM_LOG_STATS_INTERVAL": "1",   # emit metrics every second
        }
    )
)

# ## Model
# Gemma 4 26B-A4B (Mixture-of-Experts, instruction-tuned with reasoning).
# Pin a revision so the weights never change under us.
MODEL_NAME = "google/gemma-4-26B-A4B-it"
MODEL_REVISION = "47b6801b24d15ff9bcd8c96dfaea0be9ed3a0301"

# Speculative-decoding draft model (MTP assistant head).
SPECULATIVE_MODEL_NAME = "google/gemma-4-26B-A4B-it-assistant"
SPECULATIVE_MODEL_REVISION = "f188f476dc11dd5bb3014dc861529d316bce49d3"

# ## Caches
# Persisted across cold starts via Modal Volumes.
hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

# ## Performance knob
# FAST_BOOT=True  → skip Torch compilation & CUDA-graph capture (faster cold start)
# FAST_BOOT=False → full compilation (higher throughput, recommended for production)
FAST_BOOT = False

# ## App
app = modal.App("example-vllm-inference")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H200:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    secrets=[modal.Secret.from_dict({"HF_TOKEN": HF_TOKEN})],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=100)
@modal.web_server(port=VLLM_PORT, startup_timeout=10 * MINUTES)
def serve():
    """Launch the vLLM OpenAI-compatible server as a subprocess."""
    import subprocess

    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--revision", MODEL_REVISION,
        "--served-model-name", MODEL_NAME, "llm",
        "--host", "0.0.0.0",
        "--port", str(VLLM_PORT),
        "--uvicorn-log-level=info",
        "--async-scheduling",
        "--enforce-eager" if FAST_BOOT else "--no-enforce-eager",
        "--tensor-parallel-size", str(N_GPU),
        # Disable multimodal inputs to save VRAM
        "--limit-mm-per-prompt",
        f"'{json.dumps({'image': 0, 'video': 0, 'audio': 0})}'",
        # Reasoning + tool-use support
        "--enable-auto-tool-choice",
        "--reasoning-parser gemma4",
        "--tool-call-parser gemma4",
        # Speculative decoding (MTP) for better throughput at low concurrency
        "--speculative-config",
        f"'{json.dumps({'model': SPECULATIVE_MODEL_NAME, 'revision': SPECULATIVE_MODEL_REVISION, 'num_speculative_tokens': 4})}'",
    ]

    print(*cmd)
    subprocess.Popen(" ".join(cmd), shell=True)


# ## Local entrypoint (smoke test)
# Runs locally while a fresh server replica spins up on Modal.
# Usage: modal run vllm_inference.py
@app.local_entrypoint()
async def test(test_timeout=15 * MINUTES, content=None, twice=True):
    url = await serve.get_web_url.aio()

    system_prompt = {
        "role": "system",
        "content": "You are a pirate who can't help but drop sly reminders that he went to Harvard.",
    }
    if content is None:
        content = "Explain the singular value decomposition."

    messages = [
        system_prompt,
        {"role": "user", "content": content},
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            assert resp.status == 200, f"Health check failed for server at {url}"
        print(f"Health check passed ✓")

        print(f"Sending messages to {url}:", *messages, sep="\n\t")
        await _send_request(session, "llm", messages)

        if twice:
            messages[0]["content"] = "You are Jar Jar Binks."
            print(f"Sending messages to {url}:", *messages, sep="\n\t")
            await _send_request(session, "llm", messages)


async def _send_request(
    session: aiohttp.ClientSession, model: str, messages: list
) -> None:
    payload: dict[str, Any] = {
        "messages": messages,
        "model": model,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    async with session.post("/v1/chat/completions", json=payload, headers=headers) as resp:
        async for raw in resp.content:
            resp.raise_for_status()
            line = raw.decode().strip()
            if not line or line == "data: [DONE]":
                continue
            if line.startswith("data: "):
                line = line[len("data: "):]

            chunk = json.loads(line)
            assert chunk["object"] == "chat.completion.chunk"
            delta = chunk["choices"][0]["delta"]
            content = (
                delta.get("content")
                or delta.get("reasoning")
                or delta.get("reasoning_content")
            )
            if content:
                print(content, end="")
            else:
                print("\n", chunk)
    print()
