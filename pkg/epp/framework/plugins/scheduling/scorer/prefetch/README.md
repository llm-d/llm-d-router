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

```
                                  ┌──────────────────────────┐
                                  │ Inference Request         │
                                  └────────────┬──────────────┘
                                               │
                                               ▼
                          ┌──────────────────────────────────────┐
                          │ PreRequest (this plugin)             │
                          ├──────────────────────────────────────┤
                          │ 1. GetEngineKeysAndDigestsForRequest()│
                          │     ← precise-prefix-cache producer  │
                          │                                      │
                          │ 2. Build paths for ranks [0..N)      │
                          │     from operator config:            │
                          │     <rootDir>/<safeModel>_<digest>   │
                          │       _r<rank>/hhh/hh_g<G>/<hash>.bin │
                          │                                      │
                          │ 3. Submit each path to work queue    │
                          └────────────┬─────────────────────────┘
                                       │
                                       ▼
              ┌─────────── Worker Pool (M concurrent goroutines) ──────────┐
              │                                                            │
              │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ...           │
              │   │ worker 0 │  │ worker 1 │  │ worker 2 │                │
              │   └─────┬────┘  └─────┬────┘  └─────┬────┘                │
              │         │             │             │                     │
              │     read first BlockSize×BlockCount bytes from each file  │
              │         │             │             │                     │
              │     a missing file is a benign skip (block not written    │
              │     yet, or config digest/sizes don't match disk)         │
              │         │             │             │                     │
              └─────────┼─────────────┼─────────────┼─────────────────────┘
                        ▼             ▼             ▼
              ┌──────────────────────────────────────────────────┐
              │ Shared File System Mount (e.g. IBM Storage Scale) │
              │                                                   │
              │   <rootDir>/<safeModel>_<digest>_r<rank>/         │
              │     <hhh>/<hh>_g<group_idx>/<hash>.bin            │
              │                                                   │
              │   reading the head of each file triggers the      │
              │   storage system to pull the full file from       │
              │   cold object storage to the local SSD tier       │
              └──────────────────────────────────────────────────┘
```

1. The plugin calls `GetEngineKeysAndDigestsForRequest()` on the configured
   `precise-prefix-cache-producer` to obtain the full-width block-hash digests for the
   incoming request. Each digest is the content hash of one block's tokens — identical
   across all ranks for the same logical block — and names the on-disk file.
2. For each digest, the plugin builds one path per rank in the deployment
   (`Tp × Pp × Pcp × Dcp` ranks) directly from operator-supplied config: the base prefix
   is `<rootDir>/<safeModelName>_<digest>` and each rank's folder appends `_r<rank>`.
   Each rank's path points at a different on-disk file holding that rank's KV shard for
   the same logical block.
3. Each file path is submitted to a worker pool. A worker reads a configurable number of
   bytes (`BlockSize × BlockCount`) from the head of the file, which triggers the storage
   system's transparent prefetch from the cold tier to the local tier. A file that does
   not exist (the block has not been written yet, or the configured digest/sizes do not
   match what is on disk) is skipped without error.

## On-disk layout

The plugin assumes the layout written by vLLM's `llmd_fs_backend.FileMapper`:

```
<rootDir>/<safeModelName>_<digest>_r<rank>/<hhh>/<hh>_g<group_idx>/<hash>.bin
```

Where:
- `safeModelName` = `model_name` with `/` replaced by `_` (HuggingFace IDs).
- `<digest>` = the 12-hex fs-connector fingerprint (the first 12 chars of a SHA-256 over
  a JSON-canonicalized dict of vLLM-internal fields: `kv_cache_groups`, `dtype`,
  `block_size`, parallel sizes, etc.). The router cannot reliably reproduce this hash, so
  the operator reads it from the vLLM pod and supplies it via the `digest` config field.
- `<rank>` = `parallel_config.rank` for the worker that wrote the file. Ranks are iterated
  `[0, Tp×Pp×Pcp×Dcp)` from the operator-supplied parallel sizes.
- `<hhh>/<hh>` = first 5 hex chars of the block hash, sharded into two subdirectories.
- `<group_idx>` = KV cache group index, supplied via the `groupIdx` config field
  (typically `0`).
- `<hash>` = full block-hash digest (64 hex chars for SHA256-CBOR, 16 for FNV64a).
  **The same filename appears under every `_r<rank>` folder, but each rank's file holds a
  different byte content (that rank's local KV shard).**

## Architecture

The plugin uses a **concurrent worker thread pool** to prefetch multiple files in
parallel. Workers are long-lived goroutines that read from a shared work queue. A
configurable queue timeout prevents slow queues from blocking the request path.

## Configuration

The plugin is registered as `prefetch-prerequest-handler` and configured via JSON
parameters in the EPP config.

### Parameters

| Field | Type | Description |
|---|---|---|
| `engineKeysProviderPluginName` | string | Name of the `precise-prefix-cache-producer` plugin instance to use for digest retrieval |
| `kvFilePathBase` | object | KV-cache file path parameters (see below) |
| `prefetchConfig` | object | Prefetch worker pool configuration (see below) |

### `kvFilePathBase`

| Field | Type | Default | Description |
|---|---|---|---|
| `rootDir` | string | — | Root directory of the KV-cache file system mount |
| `modelName` | string | — | Model name (e.g. `meta-llama/Llama-3.1-8B-Instruct`); `/` is replaced by `_` for the on-disk folder |
| `digest` | string | — | The 12-hex fs-connector fingerprint that forms the `<safeModelName>_<digest>` base folder; read from the vLLM pod |
| `groupIdx` | int | `0` | KV cache group index for the `<hh>_g<N>` folder |
| `gpuBlocksPerFile` | int | `1` | Number of GPU blocks vLLM stores per file; the plugin emits one path per file (every Nth digest) |
| `tpSize` | int | `1` | Tensor-parallel size |
| `ppSize` | int | `1` | Pipeline-parallel size |
| `pcpSize` | int | `1` | Prefill-context-parallel size |
| `dcpSize` | int | `1` | Decode-context-parallel size |

`rootDir`, `modelName`, and `digest` are required; the plugin emits no paths until all
three are set.

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
  - name: precise-prefix-cache-producer
    type: precise-prefix-cache-producer
    parameters:
      tokenProcessorConfig:
        hashAlgorithm: sha256-cbor
        hashSeed: "10"
        blockSizeTokens: 16

  - name: kv-prefetch
    type: prefetch-prerequest-handler
    parameters:
      engineKeysProviderPluginName: precise-prefix-cache-producer
      kvFilePathBase:
        rootDir: /mnt/kv-cache
        modelName: meta-llama/Llama-3.1-8B-Instruct
        digest: 07d7b166f256
        groupIdx: 0
        gpuBlocksPerFile: 1
        tpSize: 1
        ppSize: 1
        pcpSize: 1
        dcpSize: 1
      prefetchConfig:
        enabled: true
        blockSize: 4194304
        blockCount: 3
        maxConcurrentFiles: 16
        workQueueSize: 256
        queueTimeout: 100
```

## Hash Algorithm Requirement

For digests to match the KV-cache files written by vLLM, the
`precise-prefix-cache-producer` must be configured to use the **SHA256-CBOR** hash
algorithm (`hashAlgorithm: sha256-cbor`), which matches vLLM's engine key computation.
The default FNV64a algorithm will produce different hashes and cause a mismatch.

This plugin only works with the **llm-d-fs-connector** file naming convention. File
path generation is not compatible with other KV-cache storage backends.

## Behavior notes & limitations

- **No-op until vLLM writes a block.** When fewer than `gpuBlocksPerFile` digests are
  available, no aggregated file exists on disk yet and `PreRequest` emits no paths. Files
  that are not yet present are skipped without error when a worker tries to open them.
- **Operator-supplied digest and sizes are load-bearing.** The `digest`, `groupIdx`, and
  parallel sizes must match the running vLLM deployment. A mismatch produces paths that
  never resolve, so prefetch silently no-ops (missing files are skipped). On a vLLM
  restart that shifts the fingerprint, the operator must update `digest`.
- **Single-group only.** The plugin emits paths for a single `groupIdx`. Models with
  multiple KV cache groups (sliding-window + full-attention, mamba hybrids) prefetch only
  the configured group. Multi-group support is a follow-up.
- **Single vLLM deployment per plugin instance.** One `(digest, groupIdx)` pair is
  configured per instance. Routing to multiple vLLM deployments with different parallelism
  or model configs from a single plugin instance is not supported.

## Testing

```bash
go test ./pkg/epp/framework/plugins/scheduling/scorer/prefetch/ -count=1
```

The path test file (`prefetch_prerequest_experimental_path_test.go`) exercises
config-driven base-path and full-path construction and the per-file batching by
`gpuBlocksPerFile`.
