# KV-Cache File Prefetch Plugin (Experimental)

This package implements an experimental pre-request plugin that proactively prefetches
KV-cache blocks across different storage tiers before inference requests are processed
by the GPU pod. The goal is to promote KV-cache blocks to a closer storage tier ahead
of time to reduce inference latency.

## Overview

The plugin implements the `PreRequest` interface and is invoked after a routing decision
is made but before the request is dispatched to the GPU pod. It determines the storage
locations (file paths) of KV-cache blocks that will be needed for the request and
arranges for them to be promoted to a closer storage tier.

The current implementation targets a **shared file system with transparent access to a
remote storage tier**, such as IBM Storage Scale configured to offload cold data to
remote object storage. A simple sequential read of the beginning of a KV-cache file
is sufficient to trigger the underlying storage system to promote (prefetch) the full
file from remote storage to the local file system tier.

In a future version, this could be extended to prefetch KV-cache blocks from the file
system to CPU memory on the worker node that the request is being routed to.

## How It Works

1. The plugin calls `GetEngineKeysForRequest()` on the configured
   `precise-prefix-cache` scorer to obtain the engine keys (block hashes) for the
   incoming request.
2. Engine keys are converted to file paths using the
   [llm-d-fs-connector](https://github.com/llm-d/llm-d-fs-connector) naming format:
   `<rootDir>/<model>/.../rank_N/<dtype>/hhh/hh/<16hex>.bin`
3. File paths for all tensor-parallel and pipeline-parallel ranks are computed.
4. Each file path is submitted to the worker pool, where a worker reads a configurable
   number of bytes (`BlockSize × BlockCount`) from the file to trigger the storage
   system's transparent prefetch mechanism.

## Architecture

The plugin uses a **concurrent worker thread pool** to prefetch multiple files in
parallel. Workers are long-lived goroutines that read from a shared work queue. A
configurable queue timeout prevents slow queues from blocking the request path.

```
PreRequest()
    │
    ├─ GetEngineKeysForRequest()   (precise-prefix-cache scorer)
    ├─ EngineKeysToFilePaths()     (llm-d-fs-connector format, all ranks)
    └─ workQueue ──► [worker 0]   read(BlockSize × BlockCount bytes)
                 ──► [worker 1]   read(BlockSize × BlockCount bytes)
                 ──► [worker N]   read(BlockSize × BlockCount bytes)
```

## Configuration

The plugin is registered as `prefetch-prerequest-handler` and configured via JSON
parameters in the EPP config.

### Parameters

| Field | Type | Description |
|---|---|---|
| `engineKeysProviderPluginName` | string | Name of the `precise-prefix-cache` scorer plugin instance to use for engine key retrieval |
| `kvFilePathBase` | object | KV-cache file path parameters (see below) |
| `prefetchConfig` | object | Prefetch worker pool configuration (see below) |

### `kvFilePathBase`

| Field | Type | Default | Description |
|---|---|---|---|
| `rootDir` | string | — | Root directory of the KV-cache file system mount |
| `modelParentDir` | string | `""` | Optional parent directory under `rootDir` |
| `modelName` | string | — | Model name directory segment |
| `gpuBlockSize` | int | `64` | Number of tokens per GPU block |
| `gpuBlocksPerFile` | int | `1` | Number of GPU blocks stored per file |
| `tpSize` | int | `1` | Tensor parallel size |
| `ppSize` | int | `1` | Pipeline parallel size |
| `pcpSize` | int | `1` | Pipeline checkpoint parallel size |
| `rank` | int | `0` | Rank (computed per-rank at runtime) |
| `dtype` | string | — | Data type string (e.g. `bfloat16`) |

### `prefetchConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | bool | `false` | Enable or disable the prefetch worker pool |
| `blockSize` | int64 | `4194304` (4 MiB) | Bytes to read per block from each file |
| `blockCount` | int | `3` | Number of blocks to read per file |
| `maxConcurrentFiles` | int | `16` | Number of parallel worker goroutines |
| `workQueueSize` | int | `256` | Capacity of the work queue channel |
| `queueTimeout` | int | `0` | Milliseconds to wait before skipping a file when the queue is full (0 = block indefinitely) |

### Example EPP Config Snippet

```yaml
plugins:
  - name: precise-prefix-cache-scorer
    type: precise-prefix-cache-scorer
    parameters:
      tokenProcessorConfig:
        hashAlgorithm: sha256-cbor
        hashSeed: "10"
        blockSizeTokens: 16

  - name: kv-prefetch
    type: prefetch-prerequest-handler
    parameters:
      engineKeysProviderPluginName: precise-prefix-cache-scorer
      kvFilePathBase:
        rootDir: /mnt/kv-cache
        modelName: meta-llama/Llama-3.1-8B-Instruct
        gpuBlockSize: 16
        gpuBlocksPerFile: 1
        tpSize: 1
        ppSize: 1
        pcpSize: 1
        dtype: bfloat16
      prefetchConfig:
        enabled: true
        blockSize: 4194304
        blockCount: 3
        maxConcurrentFiles: 16
        workQueueSize: 256
        queueTimeout: 100
```

## Hash Algorithm Requirement

For engine keys to match the KV-cache files written by vLLM, the `precise-prefix-cache`
scorer must be configured to use the **SHA256-CBOR** hash algorithm (`hashAlgorithm: sha256-cbor`),
which matches vLLM's engine key computation. The default FNV64a algorithm will produce
different keys and cause a mismatch.

This plugin only works with the **llm-d-fs-connector** file naming convention. File
path generation is not compatible with other KV-cache storage backends.
