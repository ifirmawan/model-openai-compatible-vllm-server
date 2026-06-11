# API Documentation

Base URL: `https://<workspace>--vibe2blog-backend-serve.modal.run`

The server is OpenAI API-compatible. Any client that works with `openai.OpenAI(base_url=...)` works here.

---

## Authentication

No API key is required. The server is protected at the network level by Modal.

---

## Endpoints

### `GET /healthz`

Liveness check. Returns `{"status": "ok"}` once vLLM is fully loaded and accepting requests. Blocks during cold start until the model is ready (up to ~2 min with cached weights).

```bash
curl https://<workspace>--vibe2blog-backend-serve.modal.run/healthz
```

**Response**

```json
{"status": "ok"}
```

> `GET /health` is intercepted by Modal's infrastructure and returns `200` with an empty body — use `/healthz` when you need a JSON response.

---

### `GET /v1/models`

Lists the model IDs registered on the server.

```bash
curl https://<workspace>--vibe2blog-backend-serve.modal.run/v1/models
```

**Response**

```json
{
  "object": "list",
  "data": [
    { "id": "google/gemma-4-26B-A4B-it", "object": "model" },
    { "id": "llm",                        "object": "model" }
  ]
}
```

Both IDs route to the same model. Use `"llm"` for a stable, model-agnostic alias.

---

### `POST /v1/chat/completions`

OpenAI-compatible chat completions endpoint. Supports both streaming (SSE) and non-streaming responses.

#### Request

| Field | Type | Required | Description |
|---|---|---|---|
| `model` | string | yes | `"llm"` or `"google/gemma-4-26B-A4B-it"` |
| `messages` | array | yes | Array of `{role, content}` objects (`system`, `user`, `assistant`) |
| `stream` | boolean | no | `true` for SSE token streaming, `false` for a single JSON response (default `false`) |
| `temperature` | float | no | Sampling temperature, 0–2 (default `1.0`) |
| `max_tokens` | integer | no | Maximum tokens to generate |
| `top_p` | float | no | Nucleus sampling probability |
| `stop` | string \| array | no | Stop sequence(s) |
| `chat_template_kwargs` | object | no | Pass `{"enable_thinking": true}` to activate Gemma's reasoning trace |

#### Non-streaming example

```bash
curl -s https://<workspace>--vibe2blog-backend-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llm",
    "stream": false,
    "messages": [
      {"role": "system", "content": "You are a concise assistant."},
      {"role": "user",   "content": "What is vLLM?"}
    ]
  }'
```

**Response**

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "model": "google/gemma-4-26B-A4B-it",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "vLLM is a fast inference engine for large language models..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 24,
    "completion_tokens": 18,
    "total_tokens": 42
  }
}
```

#### Streaming example

```bash
curl -s https://<workspace>--vibe2blog-backend-serve.modal.run/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "model": "llm",
    "stream": true,
    "messages": [
      {"role": "user", "content": "Explain gradient descent in one paragraph."}
    ]
  }'
```

Each SSE event is a `data:` line containing a JSON chunk:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"Gradient"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" descent"},"finish_reason":null}]}

data: [DONE]
```

The `delta` object may contain:

| Key | When present |
|---|---|
| `content` | Normal output token |
| `reasoning` / `reasoning_content` | Internal reasoning token (thinking mode) |

The stream ends with the literal line `data: [DONE]`.

#### Reasoning mode

Enable the model's chain-of-thought trace by adding `chat_template_kwargs`:

```json
{
  "model": "llm",
  "stream": true,
  "chat_template_kwargs": {"enable_thinking": true},
  "messages": [{"role": "user", "content": "Solve: 17 × 43"}]
}
```

Reasoning tokens arrive in `delta.reasoning` before the final `delta.content` answer.

---

## Vibe2Blog integration

Set these two environment variables in your Vibe2Blog deployment:

```bash
MODAL_VLLM_BASE_URL=https://<workspace>--vibe2blog-backend-serve.modal.run/v1
MODAL_VLLM_MODEL=llm
```

Recommended system prompt for editorial polishing:

```
You are an expert editorial assistant specialising in technical blog posts.
When given a Markdown document, polish it according to these rules:
1. Preserve every fact, figure, and technical detail exactly as given — do not invent or omit anything.
2. Preserve the Markdown structure and any YAML/TOML frontmatter unchanged.
3. Remove generic AI phrasing (e.g. 'In conclusion', 'It is worth noting', 'Delve into', 'Leverage').
4. Improve narrative flow so sentences read naturally, not like a list of disconnected facts.
5. Keep technical detail concrete — expand acronyms on first use, keep code blocks verbatim.
6. Do not add new sections, bullet points, or content that was not in the original.
Return only the polished Markdown document, with no commentary before or after it.
```

---

## Python client (openai SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://<workspace>--vibe2blog-backend-serve.modal.run/v1",
    api_key="unused",
)

response = client.chat.completions.create(
    model="llm",
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user",   "content": "What is vLLM?"},
    ],
)
print(response.choices[0].message.content)
```

---

## Limits and behaviour

| Property | Value |
|---|---|
| Max concurrent inputs | 100 |
| Cold-start timeout | 10 minutes |
| Scale-down after idle | 15 minutes |
| GPU | 1 × H200 (80 GB) |
| Model | Gemma 4 26B-A4B-it (MoE, 4B active params per token) |
| Speculative decoding | 4 draft tokens via `gemma-4-26B-A4B-it-assistant` |
