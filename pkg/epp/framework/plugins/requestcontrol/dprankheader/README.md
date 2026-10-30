# data-parallel-rank-header

A PreRequest plugin that writes the data-parallel rank of the scheduling-selected endpoint to a request header. vLLM's Rust frontend (`vllm-rs serve --data-parallel-size N`) serves N engine ranks on one port and sends a request to the rank named in the `x-data-parallel-rank` header.

The plugin acts only on rank endpoints, which share their pod's port with the pod's other ranks. The EPP splits a pod into N rank endpoints when N, the pod's rank count, is above 1. N comes from the pod's `llm-d.ai/data-parallel-size-local` annotation, or from the `--data-parallel-size-local` flag for pods without it, so pods in one pool can run different DP sizes (for example prefill with 2 ranks and decode with 4). For those endpoints it sets the header to the endpoint's data-parallel rank and replaces any value the client sent. For other endpoints it leaves the request headers unchanged.

The rank is the one vLLM numbers its engines by. Within one pod it runs from 0 to N-1. When a data-parallel group spans pods, each with its own frontend (for example an LWS group where pod `k` serves ranks `k*N` to `k*N+N-1`), set `--data-parallel-start-rank-label` to the pod label holding `k`, such as `leaderworkerset.sigs.k8s.io/worker-index`; the header then carries the global rank.

## Configuration

```yaml
- type: data-parallel-rank-header
  parameters:
    # Optional. Defaults to x-data-parallel-rank.
    headerName: x-data-parallel-rank
```

| field | meaning |
|---|---|
| `headerName` | Request header the rank is written to. EPP-owned names (stripped on the way out) and internal routing headers (`x-llm-d-prefiller-host-port` and siblings) are rejected at startup. |

The plugin reads the endpoint selected by the primary scheduling profile.

## Prefill/decode disaggregation

With the coordinator, each phase is a separate request that the EPP schedules on its own, so the prefill call carries the prefill rank and the decode call carries the decode rank.

With the P/D sidecar, one scheduling run picks the decode endpoint (primary profile) and the prefill endpoint. The header carries the decode rank. The disaggregation profile handler sends the prefill rank to the sidecar in `x-llm-d-prefiller-data-parallel-rank`, and the sidecar sets `x-data-parallel-rank` on the prefill request from it. See [Routing headers](../../../../../../docs/disaggregation.md#routing-headers).

## SGLang

SGLang's `/v1/chat/completions` and `/v1/completions` read the same `x-data-parallel-rank` header (its `routed_dp_rank`), and its metrics label each rank's series `dp_rank`, which the built-in `sglang` metrics config reads. SGLang's native `/generate` reads the rank from `x-override-routed-dp-rank`, and only when SGLang runs with `SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES=1`; set `headerName` to `x-override-routed-dp-rank` for it.

## gRPC

Envoy forwards request headers to a gRPC upstream as metadata, so the same header reaches vLLM as the `x-data-parallel-rank` metadata key.

## Example

Each model server pod runs `vllm-rs serve --data-parallel-size 2` on port 8000, selected by an InferencePool with one target port. The pods are annotated `llm-d.ai/data-parallel-size-local: "2"` (or the EPP runs with `--data-parallel-size-local=2`), so each pod becomes two endpoints, `<pod>-rank-0` and `<pod>-rank-1`, both on port 8000.

```yaml
# EndpointPickerConfig (excerpt)
plugins:
  - type: data-parallel-rank-header
```

When the EPP selects `<pod>-rank-1`, the request reaches the pod's port 8000 with `x-data-parallel-rank: 1`.
