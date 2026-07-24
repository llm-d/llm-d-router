# Pipeline Communication Protocol

This document describes the request and response formats for each stage of the coordinator pipeline. The pipeline implements the vLLM disaggregated serving protocol for multimodal inference.

> [!NOTE] 
> The encode and prefill steps support two request protocols: `/inference/v1/generate` and `/v1/chat/completions`. 
The `/inference/v1/generate` format is the preferred protocol as it naturally implements tokens-in protocol, and eliminates additional tokenization.
However, it is relatively new and may contain bugs. The `/v1/chat/completions` format is available as a fallback option, reusing the existing well-tested chat completions endpoint with an additional `tokens` field. The active protocol is controlled by the `use_openai_format` configuration (see [Request Format Configuration](#request-format-configuration)).

## Table of Contents

- [Pipeline Overview](#pipeline-overview)
- [Stage 1: replace-media-urls](#stage-1-replace-media-urls)
- [Stage 2: render](#stage-2-render)
- [Stage 3: conditional-decode](#stage-3-conditional-decode)
- [Stage 4: encode (fan-out, one per image)](#stage-4-encode-fan-out-one-per-image)
- [Stage 5: prefill](#stage-5-prefill)
- [Stage 6: decode](#stage-6-decode)
- [EPP-Phase Header and Routing](#epp-phase-header-and-routing)
- [Request Format Configuration](#request-format-configuration)
- [Completions Requests (/v1/completions)](#completions-requests-v1completions)
- [Text-Only Requests (no images)](#text-only-requests-no-images-v1chatcompletions)
- [Questions](#questions)

## Pipeline Overview

```
Client Request (/v1/chat/completions or /v1/completions)
    |
    |--- /v1/completions with token array prompt?
    |        YES --> skip to [conditional-decode]
    |
    |--- /v1/completions (text prompt)?
    |        YES --> skip replace-media-urls, go to [render]
    |
    v
[replace-media-urls] - Fan-out downloads images, converts to base64 data URIs
    |                    (skipped for /v1/completions and for /v1/chat/completions without media URLs)
    v
[render] - Tokenizes prompt, produces token_ids and per-image metadata
    |         (skipped for /v1/completions with token array prompt)
    v
[conditional-decode] - Attempts decode with token_ids;
    |                     if 412, continues pipeline; otherwise returns response
    |
    |--- /v1/completions or /v1/chat/completions without multi media content --> skip encode, go to [prefill]
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

All requests from the coordinator to workers include the `EPP-Phase` HTTP header indicating the pipeline stage (see [EPP-Phase Header and Routing](#epp-phase-header-and-routing)).

---

## Stage 1: replace-media-urls

Downloads external image URLs and replaces them with inline data URIs in the request body.

**Skipped for `/v1/completions` requests** (completions cannot contain multimedia content).

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

Sends the request body to the rendering/tokenization service. Returns the full tokenized prompt and (for chat completions with images) per-image metadata: hashes, placeholder positions, and kwargs.

The render step routes to one of two upstream paths depending on the original client request:

| Original client path     | Render endpoint                              | Skipped when                                  |
|--------------------------|----------------------------------------------|-----------------------------------------------|
| `/v1/chat/completions`   | `POST <rendering_service_address>/v1/chat/completions/render` | never                                         |
| `/v1/completions`        | `POST <rendering_service_address>/v1/completions/render`      | `prompt` is already a token array (`[]int`)   |

Batched completions prompts (`[]string` and `[][]int`) are rejected by the coordinator before the upstream call.

The two endpoints currently use **different response shapes**: chat-completions returns a single JSON object, while completions returns a one-element JSON array of that same object. The coordinator handles both; the asymmetry is documented per subsection below.

---

### 2.A `/v1/chat/completions/render`

#### Request

```
POST <rendering_service_address>/v1/chat/completions/render
Content-Type: application/json
```

Body is the full `reqCtx.Body` (with data: URIs from stage 1):

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

#### Response (single object)

The renderer echoes the request shape back with three load-bearing fields populated: `token_ids`, `features`, and `request_id`. It also fills in `sampling_params` defaults and several status fields used downstream (`priority`, `stream`, `stream_options`, `cache_salt`, `kv_transfer_params: null`). The coordinator only consumes `token_ids` and `features`; everything else is ignored at this stage.

```json
{
  "token_ids": [
    151644, 872, 198, 74785, 419, 2168, 13,
    151652,
    151655, 151655, 151655, "...(70 placeholder tokens total)...", 151655,
    151653, 151645, 198, 151644, 77091, 198
  ],
  "features": {
    "mm_hashes": {
      "image": ["2b622017706939546ca39ffbc7b610fe1fcbd4f9154d33b4ef13aaf5860c473e"]
    },
    "mm_placeholders": {
      "image": [
        {"offset": 8, "length": 70}
      ]
    },
    "kwargs_data": {
      "image": ["<base64-encoded-msgpack: pixel_values + image_grid_thw>"]
    }
  },
  "request_id": "chatcmpl-97b08c103dc4558e",
  "model": "Qwen/Qwen3-VL-2B-Instruct",
  "stream": false,
  "stream_options": null,
  "priority": 0,
  "cache_salt": null,
  "kv_transfer_params": null,
  "sampling_params": {
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "min_p": 0,
    "presence_penalty": 0,
    "frequency_penalty": 0,
    "repetition_penalty": 1,
    "max_tokens": 262060,
    "stop": [],
    "stop_token_ids": [],
    "bad_words": [],
    "output_kind": 2,
    "skip_clone": true,
    "skip_reading_prefix_cache": false
  }
}
```

##### `features` field

Three per-modality maps, each keyed by modality name (`"image"`, `"video"`, `"audio"`):

| Field                       | Element type                       | Length     | Meaning                                                                                               |
|-----------------------------|------------------------------------|------------|-------------------------------------------------------------------------------------------------------|
| `features.mm_hashes`        | `string` (hex digest)              | one per image | Stable content hash of each image. Used as the routing key for EC handoff and decode-side affinity. |
| `features.mm_placeholders`  | `{offset: int, length: int}`       | one per image | Position of each image's placeholder span inside the full `token_ids` array.                       |
| `features.kwargs_data`      | `string` (base64 of msgpack blob)  | one per image | Serialized `MultiModalKwargsItem` containing `pixel_values` and `image_grid_thw`. Encoder consumes this directly so it can skip the image preprocessor. Dominates response size (~1 MB per HD image). |

Wire-size example (the test payload above, single 200×300 JPG against Qwen3-VL-2B):

| Field                                     | Size                       |
|-------------------------------------------|----------------------------|
| `len(token_ids)`                          | 84 (7 prefix + 70 image + 7 suffix) |
| `features.mm_hashes["image"]`             | 1 entry, 64-char hex       |
| `features.mm_placeholders["image"]`       | 1 entry: `{offset: 8, length: 70}` |
| `features.kwargs_data["image"][0]`        | 1,147,128 bytes (base64)   |
| **Total response size**                   | ~1.15 MB                   |

For text-only chat completions (no `image_url` parts), `features.mm_hashes.image`, `features.mm_placeholders.image`, and `features.kwargs_data.image` are empty arrays.

#### Output (mutates RequestContext)

For the single-image example above:

- `reqCtx.TokenIDs` = the full 84-element token sequence from the response
- `reqCtx.MultimodalEntries` enriched per image:
  - `entries[0].Hash = "2b622017706939546ca39ffbc7b610fe1fcbd4f9154d33b4ef13aaf5860c473e"`
  - `entries[0].KwargsData = "<base64-encoded-msgpack blob, ~1.1 MB>"`
  - `entries[0].Placeholder = {Offset: 8, Length: 70}`

For a multi-image request the slices line up positionally:

- `entries[i].Hash = features.mm_hashes.image[i]`
- `entries[i].KwargsData = features.kwargs_data.image[i]`
- `entries[i].Placeholder = features.mm_placeholders.image[i]`

The coordinator validates that `mm_hashes.image`, `mm_placeholders.image`, and `kwargs_data.image` all have length `len(reqCtx.MultimodalEntries)`; a mismatch fails the request.

---

### 2.B `/v1/completions/render`

Used only when `reqCtx.Body["prompt"]` is a string. If the prompt is already a token array (`[]int`), the render step short-circuits and the upstream service is not called.

#### Request

```
POST <rendering_service_address>/v1/completions/render
Content-Type: application/json
```

Body is the full `reqCtx.Body`:

```json
{
  "model": "Qwen/Qwen3-VL-2B-Instruct",
  "prompt": "hello world"
}
```

#### Response (one-element array of objects)

```json
[
  {
    "request_id": "cmpl-9442e4863f950943",
    "token_ids": [14990, 1879],
    "features": null,
    "sampling_params": {
      "presence_penalty": 0.0,
      "frequency_penalty": 0.0,
      "repetition_penalty": 1.0,
      "temperature": 0.7,
      "top_p": 0.8,
      "top_k": 20,
      "min_p": 0.0,
      "stop": [],
      "stop_token_ids": [],
      "output_kind": 2,
      "skip_clone": true,
      "bad_words": [],
      "skip_reading_prefix_cache": false
    },
    "model": "Qwen/Qwen3-VL-2B-Instruct",
    "stream": false,
    "stream_options": null,
    "cache_salt": null,
    "priority": 0,
    "kv_transfer_params": null
  }
]
```

The coordinator only reads `[0].token_ids`. All other fields (`request_id`, `sampling_params`, echoed `model`, `stream`, `stream_options`, `cache_salt`, `priority`, `kv_transfer_params`) are ignored. The coordinator requires the array to have exactly one element and fails the request otherwise (it never sends a batched prompt upstream).

`features` is `null` for completions because images are not supported on this endpoint.

#### Output (mutates RequestContext)

- `reqCtx.TokenIDs` = `[14990, 1879]`
- `reqCtx.Body["prompt"]` is rewritten from the original string to the same `[]int` so downstream stages see a token-array prompt.
- `reqCtx.MultimodalEntries` is left untouched (always empty for `/v1/completions`).

When render is skipped because `prompt` is already a token array, `reqCtx.TokenIDs` is populated directly from the input array and `reqCtx.Body["prompt"]` is left as-is.

---

## Stage 3: conditional-decode

The coordinator attempts an early decode immediately after rendering. This allows the decode worker to serve the request directly if it already has the KV cache available (e.g., from a previous prefill), skipping the encode and prefill stages entirely.

The coordinator adds the `Prefer: if-available` HTTP header to signal that the decode worker should only proceed if the KV cache is already available. If it responds with 412 Precondition Failed, the pipeline continues as normal.

### Request (/v1/completions)

```
POST <gateway>/v1/completions
Content-Type: application/json
X-Request-ID: <request_id>
EPP-Phase: decode
Prefer: if-available
```

If the original `prompt` is a string, it is replaced by the `token_ids` from the render response, otherwise it will forward as is:

```json
{
  "model": "llava-v1.5-7b",
  "stream": false,
  "prompt": [1, 2345, 6789, 101, 202, 303]
}
```

> [!NOTE] 
> Check the cases of array of strings and array of array of number


### Request (/v1/chat/completions)

```
POST <gateway>/v1/chat/completions
Content-Type: application/json
X-Request-ID: <request_id>
EPP-Phase: decode
Prefer: if-available
```

The original request body is sent with a `tokens` field containing `token_ids` and `features` (without `kwargs_data`) from the render response:

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
  "tokens": {
    "token_ids": [1, 32000, 32000, 32000, 2345, 6789],
    "features": {
      "mm_hashes": {"image": ["abc123hash"]},
      "mm_placeholders": {"image": [{"offset": 1, "length": 3}]}
    }
  }
}
```

**Notes:**
- The `EPP-Phase: decode` header identifies this request as a decode attempt for routing
- The `Prefer: if-available` header signals to the decode worker that this is a conditional request - it should only proceed if the KV cache is already available
- For `/v1/completions`: the original text `prompt` is replaced with the `token_ids` array from the render response
- For `/v1/chat/completions`: the original request body is preserved and a `tokens` field is added containing `token_ids` and `features` (without `kwargs_data`)
- All other fields from the original request body (e.g., `sampling_params`, `stream`, `model`) are preserved

### Response Handling

| Status Code | Action |
|-------------|--------|
| 412 Precondition Failed | KV cache not available. Pipeline continues with encode/prefill/decode as normal. |
| 2xx (success) | Response is propagated directly to the client. Pipeline processing stops. |
| Any other error | Generic error response is propagated to the client. Pipeline processing stops. |

> [!NOTE]
> The 412 response may include additional hints in the response body or response headers (e.g., scheduling suggestions or cache locality information). These hints are not yet consumed by the coordinator and will be defined/integrated later.

---

## Stage 4: encode (fan-out, one per image)

Sends one encode request per multimodal entry. Each request contains only the BOS token plus placeholder tokens for that specific image. The encoder runs ViT and stores the result in the EC (Embedding Cache).

**Skipped for `/v1/completions` requests** (completions cannot contain multimedia content).

Two request formats are supported (see [Request Format Configuration](#request-format-configuration)). Encode requests run in parallel (configurable concurrency via `max_parallel`).

**Common notes:**
- `token_ids[0]` is always BOS (first token from render output)
- The placeholder token ID is extracted from `reqCtx.TokenIDs[entry.Placeholder.Offset]` (model-specific, opaque)
- `mm_placeholders` offset is always 1 in encode requests (right after BOS, since each request has only one image)

---

### Option A: /inference/v1/generate

#### Request (per image)

```
POST <gateway>/inference/v1/generate
Content-Type: application/json
X-Request-ID: <request_id>
EPP-Phase: encode
```

For image 0 (given `token_ids[0]=1` as BOS, `token_ids[1]=32000` as placeholder token):

```json
{
  "model": "llava-v1.5-7b",
  "token_ids": [1, 32000, 32000, 32000],
  "features": {
    "mm_hashes": {"image": ["abc123hash"]},
    "mm_placeholders": {"image": [{"offset": 1, "length": 3}]},
    "kwargs_data": {"image": ["<base64-encoded-pixel-tensor-1>"]}
  },
  "sampling_params": {"max_tokens": 1}
}
```

`model` is required by the `/inference/v1/generate` request validator and must match the served model name. Coordinator sources it from `reqCtx.Model` (populated from the inbound request body's `model` field).

#### Response

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

---

### Option B: /v1/chat/completions

#### Request (per image)

```
POST <gateway>/v1/chat/completions
Content-Type: application/json
X-Request-ID: <request_id>
EPP-Phase: encode
```

Each request contains a single image from the original message (without text content), plus a `tokens` field with per-image token_ids and features (without `kwargs_data` -- the worker extracts pixel data from the image_url directly):

For image 0:

```json
{
  "model": "llava-v1.5-7b",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "image_url",
          "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}
        }
      ]
    }
  ],
  "tokens": {
    "token_ids": [1, 32000, 32000, 32000],
    "features": {
      "mm_hashes": {"image": ["abc123hash"]},
      "mm_placeholders": {"image": [{"offset": 1, "length": 3}]}
    }
  },
  "max_tokens": 1
}
```

> [!NOTE]
> The `tokens` field is not a standard OpenAI field. It is used by EPP to prevent additional tokenization. EPP removes it from the message before forwarding to vLLM.

#### Response

```json
{
  "id": "chatcmpl-abc123",
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

---

### Response fields (both formats)

The `ec_transfer_params` map is keyed by mm_hash, with each value containing:
- `peer_host` - the host where the encoded embedding is stored
- `peer_port` - the port for the EC transfer
- `size_bytes` - size of the encoded embedding in bytes
- `nixl_agent_metadata_b64` - base64-encoded NIXL agent metadata for direct transfer

> [!NOTE]
> Format of `ec_transfer_params` depends on the EC_Connector

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

Two request formats are supported (see [Request Format Configuration](#request-format-configuration)).

**Common notes:**
- `ec_transfer_params` is a flat map keyed by mm_hash (same format as the encode response), merging all per-image entries from the encode stage
- `kv_transfer_params.do_remote_decode = true, do_remote_prefill = false` tells the prefill worker to store KV cache for remote decode
- `mm_placeholders` use the original offsets from the render response (positions in the full token sequence)

---

### Option A: /inference/v1/generate

#### Request

```
POST <gateway>/inference/v1/generate
Content-Type: application/json
X-Request-ID: <request_id>
EPP-Phase: prefill
```

```json
{
  "request_id": "req-abc-123",
  "model": "llava-v1.5-7b",
  "token_ids": [1, 32000, 32000, 32000, 32000, 32000, 32000, 2345, 6789],
  "features": {
    "mm_hashes": {"image": ["abc123hash", "def456hash"]},
    "mm_placeholders": {"image": [
      {"offset": 1, "length": 3},
      {"offset": 4, "length": 3}
    ]},
    "kwargs_data": {"image": ["<base64-encoded-pixel-tensor-1>", "<base64-encoded-pixel-tensor-2>"]}
  },
  "ec_transfer_params": {
    "abc123hash": {"peer_host": "10.0.0.1", "peer_port": 5501, "size_bytes": 2359296, "nixl_agent_metadata_b64": "TklYTA..."},
    "def456hash": {"peer_host": "10.0.0.2", "peer_port": 5502, "size_bytes": 2359296, "nixl_agent_metadata_b64": "QWdlbnQ..."}
  },
  "kv_transfer_params": {"do_remote_decode": true, "do_remote_prefill": false},
  "sampling_params": {"max_tokens": 1}
}
```

`model` is required by the `/inference/v1/generate` request validator and must match the served model name. Coordinator sources it from `reqCtx.Model` (populated from the inbound request body's `model` field).

`kwargs_data` carries the same per-image base64 tensors from the render step (same values sent to the encode stage). Each blob is a msgpack-serialized `MultiModalKwargsItem` containing both `pixel_values` and `image_grid_thw` (and any other model-specific keys). The prefill worker needs `image_grid_thw` to compute mRoPE (multimodal Rotary Position Embedding) positional encodings for the visual tokens.

> [!NOTE]
> Due to a bug in the `/inference/v1/generate` implementation, top-level `kv_transfer_params` and `ec_transfer_params` are not propagated to the engine: the endpoint reads transfer parameters only from `sampling_params.extra_args` (the top-level `ec_transfer_params` is used solely to echo back in the response). The coordinator nests both under `extra_args`:

```
POST <gateway>/inference/v1/generate
Content-Type: application/json
X-Request-ID: <request_id>
EPP-Phase: prefill
```

```json
{
  "request_id": "req-abc-123",
  "model": "llava-v1.5-7b",
  "token_ids": [1, 32000, 32000, 32000, 32000, 32000, 32000, 2345, 6789],
  "features": {
    "mm_hashes": {"image": ["abc123hash", "def456hash"]},
    "mm_placeholders": {"image": [
      {"offset": 1, "length": 3},
      {"offset": 4, "length": 3}
    ]},
    "kwargs_data": {"image": ["<base64-encoded-pixel-tensor-1>", "<base64-encoded-pixel-tensor-2>"]}
  },
  "sampling_params": {
    "max_tokens": 1,
    "extra_args": {
      "kv_transfer_params": {"do_remote_decode": true, "do_remote_prefill": false},
      "ec_transfer_params": {
        "abc123hash": {"peer_host": "10.0.0.1", "peer_port": 5501, "size_bytes": 2359296, "nixl_agent_metadata_b64": "TklYTA..."},
        "def456hash": {"peer_host": "10.0.0.2", "peer_port": 5502, "size_bytes": 2359296, "nixl_agent_metadata_b64": "QWdlbnQ..."}
      }
    }
  }
}
```

#### Response

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

---

### Option B: /v1/chat/completions

#### Request

```
POST <gateway>/v1/chat/completions
Content-Type: application/json
X-Request-ID: <request_id>
EPP-Phase: prefill
```

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
  ],
  "tokens": {
    "token_ids": [1, 32000, 32000, 32000, 32000, 32000, 32000, 2345, 6789],
    "features": {
      "mm_hashes": {"image": ["abc123hash", "def456hash"]},
      "mm_placeholders": {"image": [
        {"offset": 1, "length": 3},
        {"offset": 4, "length": 3}
      ]}
    }
  },
  "ec_transfer_params": {
    "abc123hash": {"peer_host": "10.0.0.1", "peer_port": 5501, "size_bytes": 2359296, "nixl_agent_metadata_b64": "TklYTA..."},
    "def456hash": {"peer_host": "10.0.0.2", "peer_port": 5502, "size_bytes": 2359296, "nixl_agent_metadata_b64": "QWdlbnQ..."}
  },
  "kv_transfer_params": {"do_remote_decode": true, "do_remote_prefill": false},
  "max_tokens": 1
}
```

> [!NOTE]
> The `tokens` field is not a standard OpenAI field. It is used by EPP to prevent additional tokenization. EPP removes it from the message before forwarding to vLLM.

#### Response

```json
{
  "id": "chatcmpl-abc123",
  "choices": [],
  "kv_transfer_params": {
    "block_id": "block-999",
    "peer_host": "10.0.0.42",
    "peer_port": 7777
  }
}
```

---

### /v1/completions format

For `/v1/completions` requests (no images, no encode stage):

#### Request

```
POST <gateway>/v1/completions
Content-Type: application/json
X-Request-ID: <request_id>
EPP-Phase: prefill
```

```json
{
  "request_id": "req-abc-123",
  "model": "llava-v1.5-7b",
  "prompt": [1, 2345, 6789, 101, 202, 303],
  "kv_transfer_params": {"do_remote_decode": true, "do_remote_prefill": false},
  "max_tokens": 1
}
```

No `features` or `ec_transfer_params` (no images); `prompt` contains the token array.

#### Response

```json
{
  "id": "cmpl-abc123",
  "choices": [],
  "kv_transfer_params": {
    "block_id": "block-999",
    "peer_host": "10.0.0.42",
    "peer_port": 7777
  }
}
```

---

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

### Output (mutates RequestContext)

- `reqCtx.KVTransferParams` = `{"block_id": "block-999", "peer_host": "10.0.0.42", "peer_port": 7777}`

---

## Stage 6: decode

Forwards the original client request body (enriched with `tokens`, `kv_transfer_params`, and per-image `uuid` fields) to the decode worker. Supports both streaming (SSE) and buffered responses.

### Request (/v1/chat/completions)

```
POST <gateway>/v1/chat/completions
Content-Type: application/json
X-Request-ID: <request_id>
EPP-Phase: decode
```

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
  "tokens": {
    "token_ids": [1, 32000, 32000, 32000, 32000, 32000, 32000, 2345, 6789],
    "features": {
      "mm_hashes": {"image": ["abc123hash", "def456hash"]},
      "mm_placeholders": {"image": [
        {"offset": 1, "length": 3},
        {"offset": 4, "length": 3}
      ]}
    }
  },
  "kv_transfer_params": {
    "do_remote_decode": false,
    "do_remote_prefill": true,
    "remote_engine_id": "e95b1c63-2ba6-4f26-96d0-9338d40a2560",
    "remote_block_ids": [[1]],
    "remote_request_id": "generate-tokens-550e8400-e29b-41d4-a716-446655440000",
    "remote_host": "10.130.5.242",
    "remote_port": 5557,
    "tp_size": 2
  }
}
```

> [!NOTE]
> The `kv_transfer_params` fields are connector-dependent. The example above shows the NIXL format. The fields `remote_engine_id`, `remote_block_ids`, `remote_request_id`, `remote_host`, `remote_port`, and `tp_size` are returned by the prefill worker and forwarded verbatim to the decode worker. The coordinator adds `do_remote_decode: false` and `do_remote_prefill: true`.

### Request (/v1/completions)

```
POST <gateway>/v1/completions
Content-Type: application/json
X-Request-ID: <request_id>
EPP-Phase: decode
```

```json
{
  "model": "llava-v1.5-7b",
  "stream": false,
  "prompt": [1, 2345, 6789, 101, 202, 303],
  "kv_transfer_params": {
    "do_remote_decode": false,
    "do_remote_prefill": true,
    "remote_engine_id": "e95b1c63-2ba6-4f26-96d0-9338d40a2560",
    "remote_block_ids": [[1]],
    "remote_request_id": "generate-tokens-550e8400-e29b-41d4-a716-446655440000",
    "remote_host": "10.130.5.242",
    "remote_port": 5557,
    "tp_size": 2
  }
}
```

**Notes:**
- For `/v1/chat/completions`: the original request body is preserved with a `tokens` field containing `token_ids` and `features` (without `kwargs_data`)
- For `/v1/completions`: the original text `prompt` is replaced with the `token_ids` array from the render response
- `uuid` is added to each `image_url` content part (value is the mm_hash from the render step) for multimodal cache lookup
- `image_url` retains the original base64 data URI from the replace-media-urls step so the decode worker can process images and produce the correct token sequence (matching what prefill computed)
- `kv_transfer_params` is injected at the top level of the request body for `/v1/chat/completions` and `/v1/completions`; for `/inference/v1/generate` it is nested in `sampling_params.extra_args`, since that engine reads transfer params only from there (same as the prefill request)
- `do_remote_decode: false, do_remote_prefill: true` is added by the coordinator to signal the decode worker to fetch KV from the remote prefill worker
- The `EPP-Phase: decode` header is used for routing (replaces the old `/decode/` path prefix)

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

## EPP-Phase Header and Routing

The coordinator uses the `EPP-Phase` HTTP header to identify the pipeline stage of each request sent to workers through the Inference Gateway. The gateway uses this header for routing to the correct worker pool.

| Stage             | EPP-Phase Header Value | Request Path              |
|-------------------|----------------------|---------------------------|
| Encode            | `encode`             | `/v1/chat/completions` or `/inference/v1/generate` |
| Prefill           | `prefill`            | `/v1/chat/completions`, `/v1/completions`, or `/inference/v1/generate` |
| Decode            | `decode`             | `/v1/chat/completions` or `/v1/completions` |
| Conditional-Decode| `decode`             | `/v1/chat/completions` or `/v1/completions` |

The request path matches the user's original endpoint when using OpenAI format, or `/inference/v1/generate` when using the internal format.

---

## Request Format Configuration

The `use_openai_format` setting (`pipeline.use_openai_format`, environment variable: `COORDINATOR_PIPELINE_USE_OPENAI_FORMAT`, default: `true`) controls how encode and prefill steps construct their requests. `false` (the tokens-in format) requires a `render` step in the pipeline, since render produces the token IDs the generate format sends:

- **`use_openai_format: true` (default):** The request path and body format are derived from the user's original request path at runtime. A `tokens` field is added containing `token_ids` and `features` (without `kwargs_data`).
- **`use_openai_format: false`:** Uses the internal generate format (`/inference/v1/generate`) with `token_ids` and `features` (including `kwargs_data`) directly in the body.

| User's original path | Encode format | Prefill format | Decode format |
|---------------------|---------------|----------------|---------------|
| `/v1/chat/completions` | Per-image body + `tokens` field | Original body + `tokens` + `ec_transfer_params` + `kv_transfer_params` | Original body + `tokens` + `kv_transfer_params` |
| `/v1/completions` | N/A (no images) | `{"prompt": [...], "max_tokens": 1, "kv_transfer_params": {...}, ...}` | `{"prompt": [...], "kv_transfer_params": {...}, ...}` |

When `use_openai_format: false`:

| Stage | Format |
|-------|--------|
| Encode | `{"model": "...", "token_ids": [...], "features": {..., "kwargs_data": ...}, "sampling_params": {...}}` |
| Prefill | `{"request_id": "...", "model": "...", "token_ids": [...], "features": {..., "kwargs_data": ...}, ...}` |

Note: Encode is never called for `/v1/completions` requests because completions do not support images.

---

## Completions Requests (/v1/completions)

Requests to `/v1/completions` follow a simplified pipeline:

1. **replace-media-urls**: skipped (completions cannot contain multimedia content)
2. **render**: skipped if `prompt` is already a token array (array of integers); otherwise runs to tokenize the text prompt
3. **conditional-decode**: runs normally (with `EPP-Phase: decode` header)
4. **encode**: skipped (no images)
5. **prefill**: sends request with `prompt` field containing the token array
6. **decode**: sends request with `prompt` field containing the token array + `kv_transfer_params`

---

## Text-Only Requests (no images, /v1/chat/completions)

When a `/v1/chat/completions` request contains no `image_url` parts:
- `replace-media-urls`: no-op (no downloads, no multimodal entries)
- `render`: always runs -- tokenizes the prompt and returns `token_ids` (features will be empty)
- `encode`: skipped (`MultimodalEntries` is empty)
- `prefill`: sends request with `tokens` field (token_ids only, features empty) + `kv_transfer_params`
- `decode`: sends request with `tokens` field + `kv_transfer_params`

## Questions
- Should we include ec_transfer_params into Decode request? if we want that Decoder will provide Prefill functionality for small deltas. 
