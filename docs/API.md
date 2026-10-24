# HTTP API

[README](../README.md)

`inz serve` listens on `http://127.0.0.1:8080` by default (`--host`, `--port`). It serves:
- the web UI at `/`;
- an OpenAI-compatible API under `/v1`;
- diagnostics.

`--api-key KEY` requires `Authorization: Bearer KEY` on the generation endpoints.

## `POST /v1/chat/completions`

The request is standard OpenAI chat. Supported fields:

| Field | Notes |
|---|---|
| `messages` | `system`, `developer`, `user`, `assistant` (with `reasoning_content` / `tool_calls`), `tool` |
| `tools` | OpenAI function tools, rendered into the model's tool prompt |
| `reasoning_effort` | `high` (default; the template's `xhigh`), `medium`, `low`, or `none` to disable thinking. Also `reasoning.effort` |
| `temperature`, `top_p`, `top_k`, `min_p` | defaults 1.0 / 0.95 / 20 / off, from the model's `generation_config.json` |
| `presence_penalty`, `frequency_penalty`, `repeat_penalty`, `seed` | |
| `max_tokens` / `max_completion_tokens` | default 32768 (thinking at high effort can be long) |
| `stop` | string or array |
| `stream`, `stream_options.include_usage` | Server-Sent Events |
| `speculative` | inferenZ extension: `false` disables MTP speculative decoding for this request |

The model's output is parsed into OpenAI fields:
- the `<think>…</think>` block becomes `reasoning_content`, streamed as `delta.reasoning_content`;
- the rest becomes `content`;
- `<tool_call><function=NAME><parameter=K>V</parameter>…</function></tool_call>` blocks become `tool_calls`. Each parameter is converted to the JSON type its schema declares.

The prompt is rendered by a C port of the official `chat_template.jinja`. It is byte-identical to the Jinja reference on the test suite in `tests/chat_parity.py`. Earlier assistant turns keep their reasoning, as the template does by default.

Every response, streamed or not, ends with a `timings` object (an inferenZ extension, llama.cpp style):

```json
"timings": {"prompt_n": 1203, "prompt_cached": 1180, "prompt_per_second": 41.2,
            "predicted_n": 256, "predicted_per_second": 6.9, "ttft_ms": 830,
            "drafted": 180, "accepted": 131}
```

### Prefix reuse

The engine keeps the state of the previous request. Chat clients resend the whole history, so the new request usually extends the old one; then only the new tokens are processed.

The attention caches are positional and are simply truncated. The recurrent DeltaNet state cannot be rewound token by token. It is therefore checkpointed before the last prompt token of every request, and a request that diverges after that point restarts from the checkpoint. Any other divergence recomputes the prompt.

## Tools

The engine parses tool calls but never runs them: executing code, reading files or calling services is the job of the harness that uses the API. A harness sends its tools in `tools`, receives `tool_calls`, runs them, and sends the results back as `tool` messages.

## `POST /v1/completions`

Raw text completion: `prompt` is tokenized as-is, with special tokens recognized. No chat template is applied.

## `GET /v1/models`

Returns the single model id (`qwen3.8-flash-next`).

## Diagnostics

| Endpoint | Content |
|---|---|
| `GET /v1/diag` | full JSON snapshot (see [DIAGNOSTICS.md](DIAGNOSTICS.md)) |
| `GET /metrics` | Prometheus text format |
| `POST /v1/diag/trace?tokens=N` | arm the Perfetto recorder for the next N tokens |
| `GET /v1/diag/trace` | download the last trace (`ui.perfetto.dev`) |
| `POST /tokenize` | `{"content": "..."}` → token ids |
| `GET /health` | liveness |

## Clients

Any OpenAI-compatible client works. Point it at `http://127.0.0.1:8080/v1`, with any API key unless `--api-key` is set.

```sh
curl http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Ciao!"}],"stream":true}'
```
