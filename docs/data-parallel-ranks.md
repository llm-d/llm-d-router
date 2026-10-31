# Data-parallel ranks behind one port

vLLM's Rust frontend (`vllm-rs serve --data-parallel-size N`) serves N engine ranks on one port and sends a request to the rank named in the `x-data-parallel-rank` header. The EPP can pick a rank for each request instead of leaving the choice to the frontend.

## Rank endpoints

The EPP splits a pod into N rank endpoints when N, the pod's rank count, is above 1. N comes from the pod's `llm-d.ai/data-parallel-size-local` annotation, or from the `--data-parallel-size-local` flag for pods without it, so pods in one pool can run different DP sizes (for example prefill with 2 ranks and decode with 4). The pool must have exactly one target port. The endpoints are named `<pod>-rank-<i>` and share the pod's address and port.

When the scheduler picks a rank endpoint, the EPP:

- sets `x-data-parallel-rank` to the endpoint's data-parallel rank, replacing any value the client sent;
- sends the request to that endpoint's address only, without fallback destinations, because the header applies to every destination.

Requests to endpoints without ranks keep their headers and fallbacks.

Each rank endpoint reads its own engine's metrics series (see the [metrics extractor](../pkg/epp/framework/plugins/datalayer/extractor/metrics/README.md#families-with-several-series)).

## SGLang

SGLang's `/v1/chat/completions` and `/v1/completions` read the same `x-data-parallel-rank` header (its `routed_dp_rank`), and its metrics label each rank's series `dp_rank`, which the built-in `sglang` metrics config reads. SGLang's native `/generate` reads the rank from `x-override-routed-dp-rank` instead, so the EPP's rank does not reach it.

SGLang's `/health` runs a short generation, which can take longer than the 1 second default timeout of a Kubernetes probe. A readiness probe on `/health` needs a longer `timeoutSeconds`; otherwise the pod keeps leaving and rejoining the pool, and requests get `503` with no endpoints.

## gRPC

Envoy forwards request headers to a gRPC upstream as metadata, so the same header reaches vLLM as the `x-data-parallel-rank` metadata key.

## Limitations

- **Precise prefix cache.** KV events are keyed by the pod's address, so the ranks of a pod share one index entry.
- **P/D through the sidecar.** The sidecar sends the decode rank to the prefill worker.
- **Ranks are per pod.** Each pod's ranks run from 0 to N-1, so a data-parallel group that spans pods is not addressed correctly.
- **Rank count is configured, not discovered.** The EPP does not check the annotation or the flag against the model server. A rank count above the server's sends requests to ranks the server rejects; a count below it leaves engines unscheduled.
- **No failover for rank endpoints.** A request to a rank endpoint has one destination, so when the pod fails the proxy does not retry it on another pod.
- **Health is per pod.** Pod readiness covers all of a pod's ranks. vLLM's Rust frontend reports one `/health` state for all its engines: it turns `503` after a fatal error in the frontend, and stays `200` while an engine that is still connected stops making progress. The [endpoint health filter](../pkg/epp/framework/plugins/scheduling/filter/endpointhealth/README.md) tracks each rank endpoint on its own, but its poll health does not see a stalled engine, because the frontend serves `/metrics`, and its response health counts only the configured failure statuses (`502` and `503` by default). A stalled engine's metric series keep their last values, so the scorers do not see it as busy.
- **One scrape per rank.** Each rank endpoint scrapes its pod's `/metrics` on its own, so a pod with N ranks is scraped N times per interval, each time for every engine's series.

## Example

Each model server pod runs `vllm-rs serve --data-parallel-size 2` on port 8000, selected by an InferencePool with one target port. The pods are annotated `llm-d.ai/data-parallel-size-local: "2"` (or the EPP runs with `--data-parallel-size-local=2`), so each pod becomes two endpoints, `<pod>-rank-0` and `<pod>-rank-1`, both on port 8000. When the EPP selects `<pod>-rank-1`, the request reaches the pod's port 8000 with `x-data-parallel-rank: 1`.
