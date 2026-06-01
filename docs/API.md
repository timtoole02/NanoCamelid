# NanoCamelid Local API

NanoCamelid v0.1 exposes a small local HTTP API for model discovery and
non-streaming generation. The server is intended for same-device tools and
defaults to `127.0.0.1:8080`.

Start the server:

```bash
nanocamelid serve
```

Inspect the resolved bind address, model directory, caps, and auth plan without
opening a socket:

```bash
nanocamelid serve --dry-run
```

## Defaults

| Setting | Default | Override |
| --- | --- | --- |
| Host | `127.0.0.1` | `--host <addr>` |
| Port | `8080` | `--port <port>` |
| Model directory | `/mnt/nanocamelid/models` | `--model-dir <path>` or `NANOCAMELID_MODEL_DIR` |
| Max request bytes | `65536` | `--max-request-bytes <count>` or `NANOCAMELID_MAX_REQUEST_BYTES` |
| Max input tokens | `2048` | `--max-input-tokens <count>` or `NANOCAMELID_MAX_INPUT_TOKENS` |
| Max output tokens | `256` | `--max-output-tokens <count>` or `NANOCAMELID_MAX_OUTPUT_TOKENS` |

Loopback serving can run without a token. Non-loopback binds require
`--api-key <token>` or `NANOCAMELID_API_KEY`.

When a token is configured, every endpoint requires:

```text
Authorization: Bearer <token>
```

Browser-style `OPTIONS` preflight requests are accepted without a bearer token
for the known API paths and return `204 No Content` with CORS headers:

```text
Access-Control-Allow-Origin: *
Access-Control-Allow-Methods: GET, POST, OPTIONS
Access-Control-Allow-Headers: Authorization, Content-Type
```

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Server readiness, version, model directory, and auth requirement |
| `GET` | `/v1/models` | Models found in the configured model directory |
| `POST` | `/v1/completions` | OpenAI-shaped text completion response |
| `POST` | `/v1/chat/completions` | OpenAI-shaped chat completion response |
| `GET` | `/metrics` | Prometheus-style request, response-status, uptime, and cap metrics |
| `OPTIONS` | Known paths above | Browser preflight response for local tools |

### Health

```bash
curl http://127.0.0.1:8080/health
```

Response:

```json
{
  "status": "ok",
  "version": "0.1.0",
  "model_dir": "/mnt/nanocamelid/models",
  "api_key_required": false,
  "max_request_bytes": 65536,
  "max_input_tokens": 2048,
  "max_output_tokens": 256
}
```

### Models

```bash
curl http://127.0.0.1:8080/v1/models
```

Response:

```json
{
  "object": "list",
  "model_dir": "/mnt/nanocamelid/models",
  "model_dir_exists": true,
  "data": [
    {
      "id": "Llama-3.2-1B-Instruct-Q4_0.gguf",
      "object": "model",
      "path": "/mnt/nanocamelid/models/Llama-3.2-1B-Instruct-Q4_0.gguf",
      "bytes": 123,
      "target": "llama32-1b",
      "quantization": "q4_0",
      "aliases": ["1b"]
    }
  ]
}
```

Model IDs accepted by generation endpoints are:

- `1b` or `3b` aliases for the documented Llama 3.2 default rows.
- A model filename in the configured model directory.
- A model filename stem in the configured model directory.
- An explicit `.gguf` path on the same host.

### Text Completions

```bash
curl http://127.0.0.1:8080/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"1b","prompt":"Say hello in one sentence.","max_tokens":8,"temperature":0.0}'
```

Request fields:

| Field | Required | Notes |
| --- | --- | --- |
| `model` | Yes | Alias, model filename, model stem, or explicit `.gguf` path |
| `prompt` | Yes | Non-empty string or non-empty array of strings |
| `max_tokens` | No | Positive integer, capped by `--max-output-tokens` |
| `temperature` | No | Non-negative number, defaults to `0.0` |

Response:

```json
{
  "id": "cmpl-nanocamelid",
  "object": "text_completion",
  "model": "1b",
  "choices": [
    {
      "index": 0,
      "text": "Hello from NanoCamelid.",
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 8,
    "total_tokens": 15
  }
}
```

### Chat Completions

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"1b","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":8,"temperature":0.0}'
```

Request fields:

| Field | Required | Notes |
| --- | --- | --- |
| `model` | Yes | Alias, model filename, model stem, or explicit `.gguf` path |
| `messages` | Yes | Non-empty array of `system`, `user`, or `assistant` messages with non-empty `content` |
| `max_tokens` | No | Positive integer, capped by `--max-output-tokens` |
| `temperature` | No | Non-negative number, defaults to `0.0` |

Response:

```json
{
  "id": "chatcmpl-nanocamelid",
  "object": "chat.completion",
  "model": "1b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello from NanoCamelid."
      },
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 12,
    "completion_tokens": 8,
    "total_tokens": 20
  }
}
```

### Metrics

```bash
curl http://127.0.0.1:8080/metrics
```

Response:

```text
nanocamelid_requests_total 3
nanocamelid_responses_total{status="200"} 2
nanocamelid_responses_total{status="400"} 0
nanocamelid_responses_total{status="401"} 1
nanocamelid_responses_total{status="404"} 0
nanocamelid_responses_total{status="405"} 0
nanocamelid_responses_total{status="413"} 0
nanocamelid_responses_total{status="500"} 0
nanocamelid_responses_total{status="other"} 0
nanocamelid_uptime_seconds 1.250
nanocamelid_max_request_bytes 65536
nanocamelid_max_input_tokens 2048
nanocamelid_max_output_tokens 256
```

`nanocamelid_requests_total` includes the in-flight `/metrics` request.
Response status counters report responses completed before the current metrics
body was written.

## Errors

API errors use structured JSON:

```json
{
  "error": {
    "message": "Request JSON is missing a required non-empty field.",
    "type": "invalid_request_error",
    "code": "missing_model"
  }
}
```

Common status codes and error codes:

| Status | Code | Meaning |
| --- | --- | --- |
| `400` | `bad_request` | HTTP request line is missing or invalid |
| `400` | `invalid_content_length` | `Content-Length` is not a non-negative integer |
| `400` | `missing_body` | Completion or chat request body is empty |
| `400` | `invalid_json` | Request body is not a supported JSON object |
| `400` | `missing_model` | `model` is absent or empty |
| `400` | `missing_prompt` | Completion `prompt` is absent |
| `400` | `missing_messages` | Chat `messages` is absent |
| `400` | `invalid_prompt` | Completion prompt shape is unsupported |
| `400` | `invalid_messages` | Chat message shape, role, or content is unsupported |
| `400` | `invalid_max_tokens` | `max_tokens` is not a positive integer |
| `400` | `invalid_temperature` | `temperature` is not a non-negative number |
| `400` | `input_tokens_exceeded` | Tokenized input exceeds `--max-input-tokens` |
| `400` | `output_tokens_exceeded` | Requested output exceeds `--max-output-tokens` |
| `401` | `unauthorized` | Missing or invalid bearer token |
| `404` | `model_not_found` | Requested model cannot be resolved |
| `404` | `not_found` | Endpoint path is unknown |
| `405` | `method_not_allowed` | Known endpoint path was called with an unsupported HTTP method |
| `413` | `request_too_large` | HTTP request exceeds `--max-request-bytes` |
| `500` | `generation_failed` | Model load, tokenization, or generation failed |

The v0.1 API is non-streaming. Streaming responses are planned after the first
release API contract is stable.
