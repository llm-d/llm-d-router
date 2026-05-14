# Pipeline Communication Protocol

This document describes the request and response formats for each stage of the coordinator pipeline. The pipeline implements the vLLM disaggregated serving protocol for multimodal inference.

## Pipeline Overview

```
Client Request (/v1/chat/completions)
    |
    v
[replace-media-urls] - Downloads images, converts to base64 data URIs
    |
    v
[render] - Tokenizes prompt, produces token_ids and per-image metadata
    |
    v
[encode] - Fan-out: one request per image, runs ViT encoder
    |
    v
[prefill] - Single request with full token sequence + encoder outputs
    |
    v
[decode] - Forwards to decode worker, streams response back to client
```

---

## Stage 1: replace-media-urls

Downloads external image URLs and replaces them with inline data URIs in the request body.

### Input

The original client request body (OpenAI-compatible chat completion format):

```json
{
  "model": "llava-v1.5-7b",
  "stream": false,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe these images"},
        {
          "type": "image_url",
          "image_url": {"url": "https://example.com/photo1.jpg"}
        },
        {
          "type": "image_url",
          "image_url": {"url": "https://example.com/photo2.png"}
        }
      ]
    }
  ]
}
```

### Output (mutates RequestContext)

- `reqCtx.Body["messages"]` - image URLs replaced with `data:<mime>;base64,<data>` URIs
- `reqCtx.MultimodalEntries` - populated with one entry per image:

```go
[]MultimodalEntry{
    {Index: 0, Base64Data: "<base64>", ContentType: "image/jpeg"},
    {Index: 1, Base64Data: "<base64>", ContentType: "image/png"},
}
```

---

## Stage 2: render

Sends the modified request body to the rendering/tokenization service. Returns the full tokenized prompt and per-image metadata (hashes, placeholder positions, kwargs).

### Request

```
POST <rendering_service_address>/v1/chat/completions/render
Content-Type: application/json
```

Body is the full `reqCtx.Body` (with data URIs from stage 1):

```json
{
  "model": "llava-v1.5-7b",
  "stream": false,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe these images"},
        {
          "type": "image_url",
          "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}
        },
        {
          "type": "image_url",
          "image_url": {"url": "data:image/png;base64,iVBORw0K..."}
        }
      ]
    }
  ]
}
```

### Response

```json
{
  "token_ids": [1, 32000, 32000, 32000, 32000, 32000, 32000, 2345, 6789],
  "features": {
    "mm_hashes": {
      "image": ["abc123hash", "def456hash"]
    },
    "mm_placeholders": {
      "image": [
        {"offset": 1, "length": 3},
        {"offset": 4, "length": 3}
      ]
    },
    "kwargs_data": {
      "image": ["<base64-encoded-pixel-tensor-1>", "<base64-encoded-pixel-tensor-2>"]
    }
  }
}
```

### Output (mutates RequestContext)

- `reqCtx.TokenIDs` = `[1, 32000, 32000, 32000, 32000, 32000, 32000, 2345, 6789]`
- `reqCtx.MultimodalEntries` enriched with:
  - `entries[0].Hash = "abc123hash"`
  - `entries[0].KwargsData = "<base64-encoded-pixel-tensor-1>"`
  - `entries[0].Placeholder = {Offset: 1, Length: 3}`
  - `entries[1].Hash = "def456hash"`
  - `entries[1].KwargsData = "<base64-encoded-pixel-tensor-2>"`
  - `entries[1].Placeholder = {Offset: 4, Length: 3}`

---

## Stage 3: encode (fan-out, one per image)

Sends one encode request per multimodal entry. Each request contains only the BOS token plus placeholder tokens for that specific image. The encoder runs ViT and stores the result in the EC (Embedding Cache).

### Request (per image)

```
POST <gateway>/encode<gateway_path>
Content-Type: application/json
X-Request-ID: <request_id>
```

For image 0 (given `token_ids[0]=1` as BOS, `token_ids[1]=32000` as placeholder token):

```json
{
  "token_ids": [1, 32000, 32000, 32000],
  "features": {
    "mm_hashes": {"image": ["abc123hash"]},
    "mm_placeholders": {"image": [{"offset": 1, "length": 3}]},
    "kwargs_data": {"image": ["<base64-encoded-pixel-tensor-1>"]}
  },
  "sampling_params": {"max_tokens": 1}
}
```

For image 1 (given `token_ids[0]=1` as BOS, `token_ids[4]=32000` as placeholder token):

```json
{
  "token_ids": [1, 32000, 32000, 32000],
  "features": {
    "mm_hashes": {"image": ["def456hash"]},
    "mm_placeholders": {"image": [{"offset": 1, "length": 3}]},
    "kwargs_data": {"image": ["<base64-encoded-pixel-tensor-2>"]}
  },
  "sampling_params": {"max_tokens": 1}
}
```

**Notes:**
- `token_ids[0]` is always BOS (first token from render output)
- The placeholder token ID is extracted from `reqCtx.TokenIDs[entry.Placeholder.Offset]` (model-specific, opaque)
- `mm_placeholders` offset is always 1 in encode requests (right after BOS, since each request has only one image)
- Encode requests run in parallel (configurable concurrency via `max_parallel`)

### Response (per image)

Standard GenerateResponse with `ec_transfer_params` keyed by the image's mm_hash:

```json
{
  "request_id": "req-abc-123",
  "choices": [],
  "ec_transfer_params": {
    "abc123hash": {
      "peer_host": "10.0.0.1",
      "peer_port": 5501,
      "size_bytes": 2359296,
      "nixl_agent_metadata_b64": "TklYTA..."
    }
  }
}
```

The `ec_transfer_params` map is keyed by mm_hash, with each value containing:
- `peer_host` - the host where the encoded embedding is stored
- `peer_port` - the port for the EC transfer
- `size_bytes` - size of the encoded embedding in bytes
- `nixl_agent_metadata_b64` - base64-encoded NIXL agent metadata for direct transfer

### Output (mutates RequestContext)

- `reqCtx.ECTransferParams` = ordered list matching MultimodalEntries:

```go
[]map[string]any{
    {"abc123hash": {"peer_host": "10.0.0.1", "peer_port": 5501, "size_bytes": 2359296, "nixl_agent_metadata_b64": "TklYTA..."}},
    {"def456hash": {"peer_host": "10.0.0.2", "peer_port": 5502, "size_bytes": 2359296, "nixl_agent_metadata_b64": "QWdlbnQ..."}},
}
```

---

## Stage 4: prefill

Sends a single prefill request with the full token sequence, all image metadata, and the EC transfer parameters from the encode stage. The prefill worker computes KV cache and stores it for the decode worker.

### Request

```
POST <gateway>/prefill<gateway_path>
Content-Type: application/json
X-Request-ID: <request_id>
```

```json
{
  "request_id": "req-abc-123",
  "token_ids": [1, 32000, 32000, 32000, 32000, 32000, 32000, 2345, 6789],
  "model": "llava-v1.5-7b",
  "features": {
    "mm_hashes": {"image": ["abc123hash", "def456hash"]},
    "mm_placeholders": {"image": [
      {"offset": 1, "length": 3},
      {"offset": 4, "length": 3}
    ]},
    "kwargs_data": null
  },
  "ec_transfer_params": {
    "image": [
      {"abc123hash": {"peer_host": "10.0.0.1", "peer_port": 5501, "size_bytes": 2359296, "nixl_agent_metadata_b64": "TklYTA..."}},
      {"def456hash": {"peer_host": "10.0.0.2", "peer_port": 5502, "size_bytes": 2359296, "nixl_agent_metadata_b64": "QWdlbnQ..."}}
    ]
  },
  "kv_transfer_params": {"do_remote_decode": true},
  "sampling_params": {"max_tokens": 1}
}
```

**Notes:**
- `kwargs_data` is explicitly `null` - this signals the prefill worker to fetch encoded data from the EC cache using `ec_transfer_params`
- `ec_transfer_params` is structured as per-modality: `{"image": [params_0, params_1, ...]}`
- `kv_transfer_params.do_remote_decode = true` tells the prefill worker to store KV cache for remote decode
- `mm_placeholders` use the original offsets from the render response (positions in the full token sequence)

### Response

Standard GenerateResponse with `kv_transfer_params`:

```json
{
  "request_id": "req-abc-123",
  "choices": [],
  "kv_transfer_params": {
    "block_id": "block-999",
    "peer_host": "10.0.0.42",
    "peer_port": 7777
  }
}
```

### Output (mutates RequestContext)

- `reqCtx.KVTransferParams` = `{"block_id": "block-999", "peer_host": "10.0.0.42", "peer_port": 7777}`

---

## Stage 5: decode

Forwards the original client request body (enriched with `kv_transfer_params` and per-image `uuid` fields) to the decode worker. Supports both streaming (SSE) and buffered responses.

### Request

```
POST <gateway>/decode<original_path>
Content-Type: application/json
X-Request-ID: <request_id>
```

Example for `/v1/chat/completions`:

```json
{
  "model": "llava-v1.5-7b",
  "stream": false,
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Describe these images"},
        {
          "type": "image_url",
          "image_url": null,
          "uuid": "abc123hash"
        },
        {
          "type": "image_url",
          "image_url": null,
          "uuid": "def456hash"
        }
      ]
    }
  ],
  "kv_transfer_params": {
    "block_id": "block-999",
    "peer_host": "10.0.0.42",
    "peer_port": 7777,
    "do_remote_prefill": true
  }
}
```

**Notes:**
- `uuid` is added to each `image_url` content part (value is the mm_hash from the render step)
- `image_url` is set to `null` (the decode worker doesn't need the image data, it uses uuid to reference the KV cache)
- `kv_transfer_params` is injected at the top level of the request body
- `do_remote_prefill: true` is added by the coordinator to signal the decode worker to fetch KV from the remote prefill worker
- The path uses the original client path: `/decode/v1/chat/completions` or `/decode/v1/completions`

### Response (non-streaming)

Standard OpenAI chat completion response, proxied directly to the client:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The first image shows a sunset over the ocean..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 580,
    "completion_tokens": 45,
    "total_tokens": 625
  }
}
```

### Response (streaming, `"stream": true`)

Server-Sent Events stream, proxied directly to the client:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":"The"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" first"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":" image"},"finish_reason":null}]}

...

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]

```

---

## Gateway Path Routing

The coordinator uses path prefixes to route requests through the Envoy gateway to the correct worker pool:

| Stage   | Path Format                         | Example                           |
|---------|-------------------------------------|-----------------------------------|
| Encode  | `/encode<gateway_path>`             | `/encode/inference/v1/generate`   |
| Prefill | `/prefill<gateway_path>`            | `/prefill/inference/v1/generate`  |
| Decode  | `/decode<original_client_path>`     | `/decode/v1/chat/completions`     |

The `gateway_path` is configurable per step (defaults to `/inference/v1/generate`).

---

## Text-Only Requests (no images)

When the request contains no `image_url` parts:
- `replace-media-urls`: no-op (no downloads, no multimodal entries)
- `render`: always runs - tokenizes the prompt and returns `token_ids` (features will be empty)
- `encode`: skipped (`MultimodalEntries` is empty)
- `prefill`: sends request without `features` or `ec_transfer_params`
- `decode`: sends request with `uuid` fields and `image_url: null` (no real media data), plus `kv_transfer_params`

## Questions
- Should we include ec_transfer_params into Decode request? if we want that Decoder will provide Prefill functionality for small deltas. 
