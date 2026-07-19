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

The current implementation targets a shared file system with transparent access to a
remote storage tier, such as IBM Storage Scale configured to offload cold data to
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
- `<group_idx>` = KV cache group index, iterated `[0, kvCacheGroupCount)` from the
  operator-supplied group count (`1` for standard single-group models).
- `<hash>` = full block-hash digest (64 hex chars for SHA256-CBOR, 16 for FNV64a).
  **The same filename appears under every `_r<rank>` folder, but each rank's file holds a
  different byte content (that rank's local KV shard).**

## Architecture

The plugin uses a **concurrent worker thread pool** to prefetch multiple files in
parallel. Workers are long-lived goroutines that read from a shared work queue.
Submission runs on the request's critical path, so it is **best-effort and never
blocks**: a file whose queue slot is unavailable is dropped immediately rather than
waiting for a worker. A dropped file is a warm-up miss — the backend reads it cold
later — never a request failure. There is no setting that blocks the request path.

## On-path latency overhead

Only two things run on the request's critical path: computing the block-hash digests
and enqueuing the file paths onto the work queue. Both happen in the plugin's `PreRequest`,
just before the request is handed to the proxy, and both are cheap:

- **Enqueue is a non-blocking channel send.** If a worker slot is free the path is
  queued instantly; if the queue is full the path is dropped immediately rather than
  waited on. Therefore the enqueue cost is negligible.
- **Digest computation is pure CPU work over an already-tokenized prompt.**
  Tokenization happens earlier in the token-producer; `PreRequest` only hashes the
  request's existing token blocks with SHA256-CBOR, chained per block.
  <!-- That is a few hundred small SHA-256 computations even for a ~16K-token prompt
  (256 blocks) — hardware-accelerated by CPU SHA extensions — so it completes in well
  under a millisecond. -->

The worker-pool file reads are **off-path**. The plugin only enqueues on-path; the 16
workers then read the files after handoff, concurrently with the request's backend
token generation and with the on-path processing of subsequent requests.

<!-- Those reads form a `handoff -> off-path read` phase that is never part of any
request's `received -> handoff` span, so they add no latency to the request that
triggered them. The on-path cost is therefore digest hashing plus a non-blocking
channel send: well under 1 ms per request (the enqueue itself ~0.1 ms, the whole
`PreRequest` step ~0.7 ms) against a request-admission budget of tens of
milliseconds that is dominated by tokenization.

An experiment with 100 large-prompt (~16K-token) requests measured the
EPP-internal `received -> handoff` span at ~79 ms mean with prefetch disabled and
~85 ms with it enabled. Splitting that span at the `Request handled` marker (which
fires before `PreRequest`) separates pre-scheduling work (`received -> handled`:
parse, tokenization, prefix-cache hashing, scheduling) from the `PreRequest` step
(`handled -> handoff`). Of the ~6 ms gap, only ~0.5-1 ms falls in
`handled -> handoff` — the segment where prefetch actually runs on-path, matching
the measured ~0.7 ms `PreRequest` block. The remaining ~4-6 ms falls in
`received -> handled`, where the plugin does not execute at all.

That residual is not prefetch code-path time — the plugin does not run in
`received -> handled`, which is EPP-process CPU work (parse, tokenization, block
hashing, index lookup, scheduling). The most likely cause is an indirect one:
enabling prefetch starts 16 worker goroutines in the same process, and their CPU
use (hashing, path building) and file-read syscall/GC pressure compete with the
request-handling goroutines. It is a small (~6ms or 6%) effect and, being CPU contention,
is tunable via `maxConcurrentFiles` and pod CPU sizing. The cost directly
attributable to prefetch remains the sub-millisecond on-path `PreRequest` step;
the file reads are off-path. **The prefetch plugin's on-path overhead is
negligible.** -->

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
| `kvCacheGroupCount` | int | `1` | Number of KV cache groups; paths fan out across `_g0 .. _g<N-1>`. `1` for standard single-group models; larger for hybrid/mixed-attention models |
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

### Example EPP Config Snippet

The prefetch handler depends on a `token-producer` (for tokenization) and a
`precise-prefix-cache-producer` (for block-hash digests). A representative
deployment config wiring all three together:

```yaml
pluginsConfigFile: "precise-prefix-cache-config.yaml"
pluginsCustomConfig:
  precise-prefix-cache-config.yaml: |
    apiVersion: llm-d.ai/v1alpha1 #inference.networking.x-k8s.io/v1alpha1
    kind: EndpointPickerConfig
    plugins:
      # --- v0.5.1: new required data-plane plugins ---
      - type: token-producer #tokenizer
        parameters:
          modelName: Qwen/Qwen3-8B
          vllm:
            url: "http://gaie-kv-events-ip-805c964d.lpkvc.svc.cluster.local:8000"

      - type: endpoint-notification-source

      - type: metrics-data-source

      - type: core-metrics-extractor

      # --- handler / filters ---
      - type: single-profile-handler
      - type: decode-filter

      # --- precise prefix cache: producer + scorer (official structure; replaces deprecated precise-prefix-cache-scorer) ---
      - type: precise-prefix-cache-producer
        parameters:
          tokenProcessorConfig:
            blockSize: 64
            hashAlgorithm: sha256_cbor   # must match vLLM --prefix-caching-hash-algo
            hashSeed: "10"               # must match vLLM PYTHONHASHSEED
            blockSizeTokens: 64          # must match vLLM --block-size
          speculativeIndexing: true
          indexerConfig:
            # tokenizersPoolConfig omitted: producer is tokens-only; tokens come from token-producer above
            kvBlockIndexConfig:
              enableMetrics: true
          kvEventsConfig:
            topicFilter: "kv@"
            concurrency: 4
            discoverPods: false          # static push: vLLM publishes to this EPP zmqEndpoint
            zmqEndpoint: "tcp://*:5557"

      # ... (prefix-cache-scorer and schedulingProfiles omitted)

      # --- prefetch ---
      - type: prefetch-prerequest-handler
        parameters:
          engineKeysProviderPluginName: "precise-prefix-cache-producer"
          kvFilePathBase:
            rootDir: /mnt/kv-cache-storage
            modelName: Qwen/Qwen3-8B
            digest: "07d7b166f256"
            kvCacheGroupCount: 1
            gpuBlocksPerFile: 8
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

# The prefetch rootDir must be a mounted path. Mount the shared KV-cache file
# system (e.g. IBM Storage Scale via a PVC) at kvFilePathBase.rootDir:
extraVolumeMounts:
  - name: scale-data-pvc
    mountPath: /mnt/kv-cache-storage   # must match kvFilePathBase.rootDir above

extraVolumes:
  - name: scale-data-pvc
    persistentVolumeClaim:
      claimName: afm-pvc-n3
```

## Hash Algorithm Requirement

For digests to match the KV-cache files written by vLLM, the
`precise-prefix-cache-producer` must be configured to use the **SHA256-CBOR** hash
algorithm (`hashAlgorithm: sha256_cbor`), which matches vLLM's engine key computation.
The default FNV64a algorithm will produce different hashes and cause a mismatch.

This plugin only works with the **llm-d-fs-connector** file naming convention. File
path generation is not compatible with other KV-cache storage backends.

## Behavior notes & limitations

- **No-op until vLLM writes a block.** When fewer than `gpuBlocksPerFile` digests are
  available, no aggregated file exists on disk yet and `PreRequest` emits no paths. Files
  that are not yet present are skipped without error when a worker tries to open them.
- **Coverage degrades gracefully under load.** When files arrive faster than the worker
  pool drains them (slow remote-tier reads, bursty traffic), the work queue fills and
  further files are dropped rather than queued, keeping the request path fast. The
  `skipped` count in the `prefetch submission complete` log is the queue-saturation
  signal; sustained skips mean the pool is undersized — raise `maxConcurrentFiles` or
  `workQueueSize`. There is no on-path waiting knob: a full queue always drops
  immediately to keep the request path fast.
- **Operator-supplied digest and sizes are load-bearing.** The `digest`,
  `kvCacheGroupCount`, and parallel sizes must match the running vLLM deployment. A
  mismatch produces paths that never resolve, so prefetch silently no-ops (missing files
  are skipped). On a vLLM restart that shifts the fingerprint, the operator must update
  `digest`.
- **Single vLLM deployment per plugin instance.** One `(digest, kvCacheGroupCount)` pair is
  configured per instance. Routing to multiple vLLM deployments with different parallelism
  or model configs from a single plugin instance is not supported.
- **Single model only.** `PreRequest` uses one static `kvFilePathBase` and does not read
  `request.TargetModel`. The router is otherwise multi-model — the precise-prefix-cache
  scorer routes each request within the pod pool for its target model — but this plugin
  builds every request's paths under the one configured `<safeModelName>_<digest>` prefix.
  Block-hash filenames in the SHA256-CBOR path do not encode the model, so model isolation
  on disk is carried entirely by that prefix; a request for any other model therefore
  resolves to paths that do not exist and silently no-ops. Per-model base resolution keyed
  on `request.TargetModel` is required for multi-model deployments (see Future work).
- **Path generation must match vLLM exactly.** The emitted paths have to reproduce the
  fs-connector's layout byte-for-byte: the block-hash algorithm and its inputs
  (`sha256_cbor`, `hashSeed`, `blockSizeTokens`), the `<digest>` fingerprint, the
  parallel-rank fan-out, `gpuBlocksPerFile`, and `kvCacheGroupCount` all have to equal the running
  vLLM and connector values. Any drift — a vLLM version that changes the fingerprint field
  set or the on-disk layout — produces non-matching paths and silently disables prefetch
  (missing files are skipped). The plugin cannot detect the mismatch; correctness depends
  entirely on config and connector-version alignment.

## Future work

- **Per-model base resolution.** Replace the single `kvFilePathBase` with a per-model
  lookup selected by `request.TargetModel` (the key the scorer already uses), so one
  plugin instance can serve a multi-model router. Each model's
  `(digest, tp/pp/pcp/dcp, gpuBlocksPerFile, rootDir)` is resolved independently.
- **Derive the digest at runtime instead of static config.** The `<digest>` and parallel
  sizes are read off the vLLM pod and pasted into config today. Sourcing them at runtime —
  from a vLLM API (RFC vllm-project/vllm#38147) or by reading the fs-connector's own
  `config.json` on the shared mount — removes the manual step and the drift risk called
  out in the limitations above.
- **Move prefetch execution into vLLM.** Today the router both decides what to prefetch
  and performs the file-head reads that trigger tier promotion, which forces it to
  reproduce vLLM's on-disk layout exactly. Relocating the execution into vLLM — which owns
  the connector, the digest, and the layout — would let the router signal which blocks to
  warm while vLLM resolves and promotes them, eliminating the path-matching coupling
  entirely.
- **Generalize prefetch into a framework for multiple workload patterns.** The current
  trigger is a single incoming inference request. Extending the candidate source to
  anticipated future work — batched requests, agentic / multi-turn workloads, and similar
  — would make prefetch a general KV-cache warm-up framework rather than a single-request
  pre-hook.

## Testing

```bash
go test ./pkg/epp/framework/plugins/scheduling/scorer/prefetch/ -count=1
```

The path test file (`prefetch_prerequest_experimental_path_test.go`) exercises
config-driven base-path and full-path construction and the per-file batching by
`gpuBlocksPerFile`.
