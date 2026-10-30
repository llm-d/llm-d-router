# Data-parallel ranks behind one port

vLLM's Rust frontend (`vllm-rs serve --data-parallel-size N`) serves N engine ranks on one port and sends a request to the rank named in the `x-data-parallel-rank` header. The EPP can pick a rank for each request instead of leaving the choice to the frontend.

## Rank endpoints

The EPP splits a pod into N rank endpoints when N, the pod's rank count, is above 1. N comes from the pod's `llm-d.ai/data-parallel-size-local` annotation, or from the `--data-parallel-size-local` flag for pods without it, so pods in one pool can run different DP sizes (for example prefill with 2 ranks and decode with 4). The pool must have exactly one target port. The endpoints are named `<pod>-rank-<i>` and share the pod's address and port.

When the scheduler picks a rank endpoint, the EPP:

- sets `x-data-parallel-rank` to the endpoint's data-parallel rank, replacing any value the client sent;
- sends the request to that endpoint's address only, without fallback destinations, because the header applies to every destination.

Requests to endpoints without ranks keep their headers and fallbacks.

Each rank endpoint reads its own engine's metrics series (see the [metrics extractor](../pkg/epp/framework/plugins/datalayer/extractor/metrics/README.md#families-with-several-series)) and, with the [precise prefix cache](../pkg/epp/framework/plugins/requestcontrol/dataproducer/preciseprefixcache/README.md), its own KV events.

## Global ranks across pods

Within one pod the ranks run from 0 to N-1. When a data-parallel group spans pods, each with its own frontend (for example an LWS group where pod `k` serves ranks `k*N` to `k*N+N-1`), set `--data-parallel-start-rank-label` to the pod label holding `k`, such as `leaderworkerset.sigs.k8s.io/worker-index`. The header, the metrics rank filter and the KV-event ports then use the global rank, which is how vLLM numbers its engines.

vLLM's Python API server (`vllm serve`) under hybrid or external load balancing reads `x-data-parallel-rank` as an index into its own engines and rejects values of `data_parallel_size_local` or more ([vllm-project/vllm#64801](https://github.com/vllm-project/vllm/issues/64801)). The start-rank label therefore works with the vLLM Rust frontend only. Single-pod groups work with both frontends, since their global and local ranks are equal.

Keep the KV-event and P2P base ports the same on every pod; see the wide-EP notes under [KV Connector](disaggregation.md#kv-connector---kv-connector).

## Prefill/decode disaggregation

With the coordinator, each phase is a separate request that the EPP schedules on its own, so the prefill call carries the prefill rank and the decode call carries the decode rank.

With the P/D sidecar, one scheduling run picks the decode endpoint (primary profile) and the prefill endpoint. The header carries the decode rank. The disaggregation profile handler sends the prefill rank to the sidecar in `x-llm-d-prefiller-data-parallel-rank`, and the sidecar sets `x-data-parallel-rank` on the prefill request from it. With the P2P offloading connector, the sidecar also adds the prefill rank to the prefiller's P2P port. See [Routing headers](disaggregation.md#routing-headers).

Not covered:

- Mooncake with `--data-parallel-start-rank-label`: vLLM registers Mooncake engines under the pod-local rank, while the EPP sends the global one.
- The P2P cache-source pull: `x-llm-d-kv-cache-source-host-port` carries no rank.

## SGLang

SGLang's `/v1/chat/completions` and `/v1/completions` read the same `x-data-parallel-rank` header (its `routed_dp_rank`), and its metrics label each rank's series `dp_rank`, which the built-in `sglang` metrics config reads. SGLang's native `/generate` reads the rank from `x-override-routed-dp-rank` instead, so the EPP's rank does not reach it.

With one HTTP server for a group that spans nodes, that server's `/metrics` covers only its own node's ranks. Its `/v1/loads?format=prometheus` covers every rank and can be the metrics source.

## gRPC

Envoy forwards request headers to a gRPC upstream as metadata, so the same header reaches vLLM as the `x-data-parallel-rank` metadata key.

## Example

Each model server pod runs `vllm-rs serve --data-parallel-size 2` on port 8000, selected by an InferencePool with one target port. The pods are annotated `llm-d.ai/data-parallel-size-local: "2"` (or the EPP runs with `--data-parallel-size-local=2`), so each pod becomes two endpoints, `<pod>-rank-0` and `<pod>-rank-1`, both on port 8000. When the EPP selects `<pod>-rank-1`, the request reaches the pod's port 8000 with `x-data-parallel-rank: 1`.
