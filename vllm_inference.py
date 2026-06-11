# Run OpenAI-compatible LLM inference with Gemma and vLLM
#
# Deploy:   modal deploy vllm_inference.py
# Test run: modal run vllm_inference.py

import json
import os
from typing import Any

import aiohttp
import modal

# ## Container image
# Installed on top of the CUDA 12.9 base image provided by Modal.
vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "httpx==0.28.1")
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
app = modal.App("vibe2blog-backend")

N_GPU = 1
MINUTES = 60  # seconds
VLLM_PORT = 8000


@app.function(
    image=vllm_image,
    gpu=f"H200:{N_GPU}",
    scaledown_window=15 * MINUTES,
    timeout=10 * MINUTES,
    # HF_TOKEN is stored in a Modal named secret.
    # Create it once with: modal secret create huggingface HF_TOKEN=<your-token>
    secrets=[modal.Secret.from_name("huggingface")],
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def serve():
    """FastAPI proxy in front of vLLM — /health returns JSON; all other routes stream through."""
    import subprocess
    import fastapi
    import fastapi.responses
    import httpx

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
        "--limit-mm-per-prompt",
        f"'{json.dumps({'image': 0, 'video': 0, 'audio': 0})}'",
        "--enable-auto-tool-choice",
        "--reasoning-parser gemma4",
        "--tool-call-parser gemma4",
        "--speculative-config",
        f"'{json.dumps({'model': SPECULATIVE_MODEL_NAME, 'revision': SPECULATIVE_MODEL_REVISION, 'num_speculative_tokens': 4})}'",
    ]
    print(*cmd)
    subprocess.Popen(" ".join(cmd), shell=True)

    proxy = fastapi.FastAPI()
    vllm_base = f"http://0.0.0.0:{VLLM_PORT}"

    @proxy.get("/health")
    async def health():
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{vllm_base}/health")
            r.raise_for_status()
        return {"status": "ok"}

    @proxy.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"])
    async def passthrough(path: str, request: fastapi.Request):
        body = await request.body()
        headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}
        client = httpx.AsyncClient(timeout=None)
        r = await client.send(
            client.build_request(
                method=request.method,
                url=f"{vllm_base}/{path}",
                headers=headers,
                content=body,
                params=request.query_params,
            ),
            stream=True,
        )
        resp_headers = {
            k: v for k, v in r.headers.items()
            if k.lower() not in ("transfer-encoding", "content-encoding")
        }

        async def body_iter():
            async for chunk in r.aiter_raw():
                yield chunk
            await r.aclose()
            await client.aclose()

        return fastapi.responses.StreamingResponse(
            body_iter(),
            status_code=r.status_code,
            headers=resp_headers,
        )

    return proxy


# ## Local entrypoint (smoke test)
# Runs locally while a fresh server replica spins up on Modal.
# Usage: modal run vllm_inference.py
@app.local_entrypoint()
async def test(test_timeout=15 * MINUTES):
    url = await serve.get_web_url.aio()

    system_prompt = {
        "role": "system",
        "content": (
            "You are an expert editorial assistant specialising in technical blog posts. "
            "When given a Markdown document, polish it according to these rules:\n"
            "1. Preserve every fact, figure, and technical detail exactly as given — do not invent or omit anything.\n"
            "2. Preserve the Markdown structure and any YAML/TOML frontmatter unchanged.\n"
            "3. Remove generic AI phrasing (e.g. 'In conclusion', 'It is worth noting', 'Delve into', 'Leverage').\n"
            "4. Improve narrative flow so sentences read naturally, not like a list of disconnected facts.\n"
            "5. Keep technical detail concrete — expand acronyms on first use, keep code blocks verbatim.\n"
            "6. Do not add new sections, bullet points, or content that was not in the original.\n"
            "Return only the polished Markdown document, with no commentary before or after it."
        ),
    }

    sample_post = """\
---
title: "Deploying Gemma 4 on Modal"
date: 2026-06-11
tags: [llm, modal, vllm]
---

## Introduction

In this post we will delve into the process of deploying Gemma 4 26B-A4B-it
on Modal using vLLM. It is worth noting that this model leverages a
Mixture-of-Experts architecture, which means it activates only 4B parameters
per token despite having 26B total parameters.

## Setup

Firstly, you need to install the required dependencies. The key dependency is
vllm==0.21.0. You also need a Hugging Face token to download the gated weights.

## Conclusion

In conclusion, deploying Gemma 4 on Modal is straightforward and cost-effective.
"""

    messages = [
        system_prompt,
        {"role": "user", "content": f"Please polish the following post:\n\n{sample_post}"},
    ]

    async with aiohttp.ClientSession(base_url=url) as session:
        print(f"Running health check for server at {url}")
        async with session.get("/health", timeout=test_timeout - 1 * MINUTES) as resp:
            assert resp.status == 200, f"Health check failed for server at {url}"
        print("Health check passed ✓")

        print(f"Sending blog-polish request to {url}:")
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
