# Data-parallel ranks behind one port

vLLM's Rust frontend (`vllm-rs serve --data-parallel-size N`) serves N engine ranks on one port and sends a request to the rank named in the `x-data-parallel-rank` header. The EPP can pick a rank for each request instead of leaving the choice to the frontend.

A model server that listens on a separate port for each rank needs no rank header. See [One port for each rank](#one-port-for-each-rank).

## Rank endpoints

The EPP splits a pod into N rank endpoints when N, the pod's rank count, is above 1. N comes from the pod's `llm-d.ai/data-parallel-size-local` annotation. Pods in one pool can thus run different DP sizes (for example prefill with 2 ranks and decode with 4 with coordinator P/D; sidecar P/D is covered in Limitations below). The pool must have exactly one target port. N is at most 64, and a pod with a larger or malformed value gets no endpoints. The endpoints are named `<pod>-rank-<i>`, where `i` is the rank, and share the pod's address and port.

With file discovery, list one endpoint for each rank. Give all of them the address and port of the model server. In each entry, set `rankIndex` to the rank and `rankByHeader` to `true`:

```yaml
endpoints:
  - name: decode-0-rank-0
    address: 10.0.0.2
    port: "8000"
    rankIndex: 0
    rankByHeader: true
  - name: decode-0-rank-1
    address: 10.0.0.2
    port: "8000"
    rankIndex: 1
    rankByHeader: true
```

`rankIndex` alone does not make a rank endpoint. The ports of a multi-port model server also have a `rankIndex`, and they get no rank header.

When the scheduler picks a rank endpoint, the EPP:

- sets `x-data-parallel-rank` to the endpoint's data-parallel rank, replacing any value the client or a PreRequest plugin set;
- sends the request to that endpoint's address only, without fallback destinations, because the header applies to every destination.

Requests to endpoints without ranks keep their headers and their fallbacks without ranks. The EPP drops a rank endpoint from the fallbacks of such a request, because it sets no rank for it.

With the [precise prefix cache](../pkg/epp/framework/plugins/requesthandling/dataproducer/preciseprefixcache/README.md), each rank endpoint reads its own KV events.

## Global ranks across pods

Within one pod the ranks run from 0 to N-1. When a data-parallel group spans pods, each with its own frontend (for example an LWS group where pod `k` serves ranks `k*N` to `k*N+N-1`), set `--data-parallel-start-rank-label` to the pod label holding `k`, such as `leaderworkerset.sigs.k8s.io/worker-index`. The header, the metrics rank filter and the KV-event ports then use the global rank, which is how vLLM numbers its engines.

vLLM's Python API server (`vllm serve`) under hybrid or external load balancing reads `x-data-parallel-rank` as an index into its own engines and rejects values of `data_parallel_size_local` or more ([vllm-project/vllm#64801](https://github.com/vllm-project/vllm/issues/64801)). The start-rank label therefore works with the vLLM Rust frontend only. Single-pod groups work with both frontends, since their global and local ranks are equal.

Keep the KV-event and P2P base ports the same on every pod; see the wide-EP notes under [KV Connector](disaggregation.md#kv-connector---kv-connector).

## SGLang

SGLang's `/v1/chat/completions` and `/v1/completions` read the same `x-data-parallel-rank` header (its `routed_dp_rank`). SGLang's native `/generate` reads `x-override-routed-dp-rank` instead, and `sglang.api.v1` gRPC reads `routed_dp_rank` from the request body, so the EPP's `x-data-parallel-rank` header does not reach either route.

With one HTTP server for a group that spans nodes, that server's `/metrics` covers only its own node's ranks. Its `/v1/loads?format=prometheus` covers every rank and can be the metrics source.

SGLang's `/health` runs a short generation, which can take longer than the 1 second default timeout of a Kubernetes probe. A readiness probe on `/health` needs a longer `timeoutSeconds`; otherwise the pod keeps leaving and rejoining the pool, and requests get `503` with no endpoints.

## gRPC

Envoy forwards request headers to a gRPC upstream as metadata, so the same header reaches vLLM as the `x-data-parallel-rank` metadata key.

## Limitations

- **Metrics are per pod.** Every rank endpoint of a pod reads the pod's `/metrics` without filtering by rank, so a family with one series per engine gives each rank the same value.
- **P/D through the sidecar.** The sidecar sends the decode rank to the prefill worker.
- **Rank count and start rank are configured, not discovered.** The EPP does not check the annotation, the file entries, or the start-rank label against the model server. A rank count above the server's sends requests to ranks the server rejects; a count below it leaves engines unscheduled.
- **No failover for rank endpoints.** A request to a rank endpoint has one destination, so when the pod fails the proxy does not retry it on another pod.
- **Health is per pod.** Pod readiness covers all of a pod's ranks. vLLM's Rust frontend reports one `/health` state for all its engines: it turns `503` after a fatal error in the frontend, and stays `200` while an engine that is still connected stops making progress. The [endpoint health filter](../pkg/epp/framework/plugins/scheduling/filter/endpointhealth/README.md) tracks each rank endpoint on its own, but its poll health does not see a stalled engine, because the frontend serves `/metrics`, and its response health counts only the configured failure statuses (`502` and `503` by default).
- **One scrape per rank.** Each rank endpoint scrapes its pod's `/metrics` on its own, so a pod with N ranks is scraped N times per interval, each time for every engine's series.
- **KV events from engines on another pod.** When one frontend serves engines on other pods (vLLM internal load balancing, SGLang with one HTTP server per group), those engines publish KV events on their own pods, which the EPP does not know. The precise prefix cache then sees only the frontend pod's ranks. vLLM v0.32.0 reports each engine's reachable publisher in `/kv_event_sources` when the publisher is configured with `tcp://*:0`; the EPP does not read it. SGLang has no per-node frontend ([sgl-project/sglang#47657](https://github.com/sgl-project/sglang/issues/47657)).
- **Mooncake with `--data-parallel-start-rank-label`.** vLLM registers Mooncake engines under the pod-local rank, while the EPP sends the global one.
- **Global ranks with one port for each rank.** The EPP ignores `--data-parallel-start-rank-label` when a pod uses multiple target ports. Every pod gets local rank offsets from `0`. When a data-parallel group spans pods, vLLM numbers its KV-event ports by global rank instead.

## Example

Each model server pod runs `vllm-rs serve --data-parallel-size 2` on port 8000, selected by an InferencePool with one target port. The pods are annotated `llm-d.ai/data-parallel-size-local: "2"`, so each pod becomes two endpoints, `<pod>-rank-0` and `<pod>-rank-1`, both on port 8000. When the EPP selects `<pod>-rank-1`, the request reaches the pod's port 8000 with `x-data-parallel-rank: 1`.

## One port for each rank

When the model server listens on a separate port for each rank, the port selects the rank, and the EPP sends no rank header. List the port of each rank in the target ports of the pool, in rank order:

- With an `InferencePool`: `spec.targetPorts`, which takes 1 to 8 ports.
- Without an `InferencePool` (`--endpoint-selector`): `--endpoint-target-ports`, which takes 1 to 8 ports.
- With the Helm charts: `router.modelServers.targetPorts`, which sets the one of the two that applies. See [Data-parallel ranks](../config/charts/README.md#data-parallel-ranks).

The EPP makes one endpoint for each target port of a pod. The endpoint is named `<pod>-rank-<i>`, where `i` is the position of the port in the list, and the EPP scrapes its metrics on its own port. These plugins add `i` to a base port to find another listener of the same rank, so the order of the list must be the rank order of the model server:

- The [precise prefix cache](../pkg/epp/framework/plugins/requesthandling/dataproducer/preciseprefixcache/README.md), for the KV-event sockets.
- The [`p2p-source-producer`](../pkg/epp/framework/plugins/requesthandling/dataproducer/p2psource/README.md), for the P2P port.
- The [tokenizer](../pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer/README.md), for the render port of a port rule.

With file discovery, list one entry for each port, with the rank in `rankIndex` and without `rankByHeader`. See [Endpoints file format](discovery.md#endpoints-file-format).
