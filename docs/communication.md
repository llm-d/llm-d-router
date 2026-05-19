# Pipeline Communication Protocol

This document describes the request and response formats for each stage of the coordinator pipeline. The pipeline implements the vLLM disaggregated serving protocol for multimodal inference.

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Stage 1: replace-media-urls](#stage-1-replace-media-urls)
- [Stage 2: render](#stage-2-render)
- [Stage 3: conditional-decode](#stage-3-conditional-decode)
- [Stage 4: encode (fan-out, one per image)](#stage-4-encode-fan-out-one-per-image)
- [Stage 5: prefill](#stage-5-prefill)
- [Stage 6: decode](#stage-6-decode)
- [Gateway Path Routing](#gateway-path-routing)
- [Text-Only Requests (no images)](#text-only-requests-no-images)
- [Questions](#questions)

## Pipeline Overview

```
Client Request (/v1/chat/completions or /v1/completions)
    |
    v
[replace-media-urls] - Downloads images, converts to base64 data URIs
    |
    v
[render] - Tokenizes prompt, produces token_ids and per-image metadata
    |
    v
[conditional-decode] - Attempts decode with token_ids;
    |                     if 412, continues pipeline; otherwise returns response
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

## Stage 3: conditional-decode

The coordinator attempts an early decode immediately after rendering. This allows the decode worker to serve the request directly if it already has the KV cache available (e.g., from a previous prefill), skipping the encode and prefill stages entirely.

The coordinator adds the `Prefer: if-available` HTTP header to signal that the decode worker should only proceed if the KV cache is already available. If it responds with 412 Precondition Failed, the pipeline continues as normal.

### Request (/v1/completions)

```
POST <gateway>/decode/v1/completions
Content-Type: application/json
X-Request-ID: <request_id>
Prefer: if-available
```

Since completions requests do not support multimedia content, the render response will not include `mm_placeholders` or `kwargs_data` -- only `token_ids` are produced. The original `prompt` field is replaced by the `token_ids` from the render response:

```json
{
  "model": "llava-v1.5-7b",
  "stream": false,
  "prompt": [1, 2345, 6789, 101, 202, 303]
}
```

### Request (/v1/chat/completions)

```
POST <gateway>/decode/v1/chat/completions
Content-Type: application/json
X-Request-ID: <request_id>
Prefer: if-available
```

The original request body is sent as-is, with `token_ids` from the render response added:

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
        }
      ]
    }
  ],
  "token_ids": [1, 32000, 32000, 32000, 2345, 6789]
}
```

**Notes:**
- The `Prefer: if-available` header signals to the decode worker that this is a conditional request -- it should only proceed if the KV cache is already available
- For `/v1/completions`: the original text `prompt` is replaced with the `token_ids` array from the render response
- For `/v1/chat/completions`: the original request body is preserved and `token_ids` is added as an additional top-level field
- All other fields from the original request body (e.g., `sampling_params`, `stream`, `model`) are preserved

### Response Handling

| Status Code | Action |
|-------------|--------|
| 412 Precondition Failed | KV cache not available. Pipeline continues with encode/prefill/decode as normal. |
| 2xx (success) | Response is propagated directly to the client. Pipeline processing stops. |
| Any other error | Generic error response is propagated to the client. Pipeline processing stops. |

**Note:** The 412 response may include additional hints in the response body or response headers (e.g., scheduling suggestions or cache locality information). These hints are not yet consumed by the coordinator and will be defined/integrated later.

---

## Stage 4: encode (fan-out, one per image)

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
  "request_id": "generate-tokens-abc123",
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

## Stage 5: prefill

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
    "kwargs_data": {"image": ["<base64-encoded-pixel-tensor-1>", "<base64-encoded-pixel-tensor-2>"]}
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
- `kwargs_data` carries the same per-image base64 tensors from the render step (same values sent to the encode stage). Each blob is a msgpack-serialized `MultiModalKwargsItem` containing both `pixel_values` and `image_grid_thw` (and any other model-specific keys). The prefill worker needs `image_grid_thw` to compute mRoPE (multimodal Rotary Position Embedding) positional encodings for the visual tokens. The embeddings themselves encode what each patch contains; `image_grid_thw` tells the model where each patch sits spatially (temporal, height, width grid dimensions).
- `ec_transfer_params` is structured as per-modality: `{"image": [params_0, params_1, ...]}`
- `kv_transfer_params.do_remote_decode = true` tells the prefill worker to store KV cache for remote decode
- `mm_placeholders` use the original offsets from the render response (positions in the full token sequence)

### Optimization: avoid sending pixel data to prefill

Currently the full `kwargs_data` blobs (containing both `pixel_values` and `image_grid_thw`) are forwarded to the prefill worker. The prefill worker only needs `image_grid_thw` for mRoPE -- the `pixel_values` are redundant since the encoder already consumed them. For large images, the pixel tensors dominate the payload size, so stripping them would significantly reduce the data sent to prefill.

**Required changes:**

1. **vLLM render endpoint** (`vllm/entrypoints/openai/render/serving.py`): return `image_grid_thw` as a separate top-level field in the render response, alongside `kwargs_data`. The render step already computes it during image preprocessing (`get_image_grid_thw()` in the vision processor). Example response:
   ```json
   {
     "token_ids": [1, 32000, 32000, 32000, ...],
     "features": {
       "mm_hashes": {"image": ["abc123hash", "def456hash"]},
       "mm_placeholders": {"image": [{"offset": 1, "length": 3}, {"offset": 4, "length": 3}]},
       "kwargs_data": {"image": ["<full-msgpack-blob-1>", "<full-msgpack-blob-2>"]},
       "image_grid_thw": {"image": [[1, 24, 24], [1, 16, 16]]}
     }
   }
   ```

2. **vLLM prefill worker**: accept `image_grid_thw` directly in the features dict (as plain JSON arrays) instead of extracting it from the msgpack `kwargs_data` blob.

3. **Coordinator render step** (`pkg/steps/render.go`): parse `image_grid_thw` from the render response and store it per `MultimodalEntry`.

4. **Coordinator prefill step** (`pkg/steps/prefill.go`): send `image_grid_thw` instead of `kwargs_data` in the prefill request features:
   ```json
   "features": {
     "mm_hashes": {"image": ["abc123hash", "def456hash"]},
     "mm_placeholders": {"image": [{"offset": 1, "length": 3}, {"offset": 4, "length": 3}]},
     "image_grid_thw": {"image": [[1, 24, 24], [1, 16, 16]]}
   }
   ```

5. **Coordinator encode step** (`pkg/steps/encode.go`): no change -- encode continues to send the full `kwargs_data` (pixel values needed for ViT).

### Response

Standard GenerateResponse with `kv_transfer_params`:

```json
{
  "request_id": "generate-tokens-abc123",
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

## Stage 6: decode

Forwards the original client request body (enriched with `token_ids`, `kv_transfer_params`, and per-image `uuid` fields) to the decode worker. Supports both streaming (SSE) and buffered responses.

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
          "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."},
          "uuid": "abc123hash"
        },
        {
          "type": "image_url",
          "image_url": {"url": "data:image/jpeg;base64,iVBORw0K..."},
          "uuid": "def456hash"
        }
      ]
    }
  ],
  "token_ids": [1, 32000, 32000, 32000, 32000, 32000, 32000, 2345, 6789, ...],
  "kv_transfer_params": {
    "block_id": "block-999",
    "peer_host": "10.0.0.42",
    "peer_port": 7777,
    "do_remote_prefill": true
  }
}
```

Example for `/v1/completions`:

```json
{
  "model": "llava-v1.5-7b",
  "stream": false,
  "prompt": [1, 2345, 6789, 101, 202, 303],
  "kv_transfer_params": {
    "block_id": "block-999",
    "peer_host": "10.0.0.42",
    "peer_port": 7777,
    "do_remote_prefill": true
  }
}
```

**Notes:**
- For `/v1/chat/completions`: `token_ids` is added as an additional top-level field alongside the original request body
- For `/v1/completions`: the original text `prompt` is replaced with the `token_ids` array from the render response
- `uuid` is added to each `image_url` content part (value is the mm_hash from the render step) for multimodal cache lookup
- `image_url` retains the original base64 data URI from the replace-media-urls step so the decode worker can process images and produce the correct token sequence (matching what prefill computed)
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
| Encode  | `/encode/<gateway_path>`             | `/encode/inference/v1/generate`   |
| Prefill | `/prefill/<gateway_path>`            | `/prefill/inference/v1/generate`  |
| Decode  | `/decode/<original_client_path>`     | `/decode/v1/chat/completions` or  `/decode/v1/completions`  |

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
