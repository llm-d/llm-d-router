# P2P Source Producer Plugin

**Type:** `p2p-source-producer`

Selects a peer that can supply the most cached prompt prefix and writes it into the
request body as `kv_transfer_params.remote_kv_source`, so the pod computing the prefix
pulls it over the vLLM `OffloadingConnector` P2P tier instead of recomputing it. The same
peer goes on the `x-llm-d-kv-cache-source-p2p-host-port` request header, and the prefill
pod's P2P tier address on `x-llm-d-kv-cache-prefill-p2p-host-port`. Source selection runs in
the `DataProducer` phase before scheduling; the body field and the headers are written
in `PreRequest`, after the computing endpoint is known.

The plugin consumes per-endpoint `PrefixCacheMatchInfo` from an approximate or
precise prefix-cache producer. It samples among endpoints within one block of
the largest eligible prefix, weighted by `1/(1+waiting queue)` and a
request-ID hash. This spreads pulls across replicas that hold nearly identical
prefixes. After scheduling, the source is written only when the sampled source
out-caches the computing endpoint by at least `minCachedTokenDelta` tokens. The
computing endpoint is the prefill profile's target when that profile has one, and the
primary profile's target otherwise. A source that is itself one of that profile's
selected endpoints is never written: with `maxNumOfEndpoints` above `1` the routing
layer may serve the request from any of them.

## Outputs

- `kv_transfer_params.remote_kv_source` in the request body, when the source is kept:
  a fresh `kv_request_id`, the source's `remote_host` and its P2P tier `remote_port`.
  The field is the form vLLM reads. It is not written when the body already carries
  `kv_transfer_params.remote_prefiller`: such a body is a decode request that already
  pulls the prompt from a prefill pod, so a source next to it would be pointless.
- `x-llm-d-kv-cache-source-p2p-host-port` on the request headers, in the same step as the
  body field, as the source's `host:P2P port`.
- `x-llm-d-kv-cache-prefill-p2p-host-port` on the request headers, whenever the prefill
  profile has a target, as the prefill pod's `host:P2P port`. The sidecar builds the
  decode request's `remote_prefiller` from it. The prefill pod's serving address,
  `x-llm-d-prefiller-host-port`, is the `disagg-profile-handler`'s output and is not
  written here.
- The decision for the request (`P2PSourceDecision`: the `outcome` and `endpoint_role` of
  `llm_d_epp_p2p_source_decisions_total`) as a request attribute, read by a
  `prefix-cache-affinity-filter` whose `p2pSourceProducerName` names this instance.

A P2P tier port is `p2pConnectorPort` plus the endpoint's data parallel rank, the
position of its port in the pool's target ports. See
[the sidecar KV connector flag](../../../../../../../docs/disaggregation.md#kv-connector---kv-connector) for the vLLM side.

## Reusable Prefix Floor

Per-tier cache data distinguishes the source's pull-servable CPU blocks from
its GPU-only blocks. For a sampled source with a confirmed contiguous CPU
prefix, the plugin publishes a name-bound `ReusablePrefixTokens` floor for
scheduling consumers. Let `S` be that prefix length and `D` be
`minCachedTokenDelta`:

```text
G = max(0, S - D + 1)
```

The attribute is omitted when `G` is zero.

The [`ReusablePrefixTokens` type](../../../datalayer/attribute/prefix/data_types.go)
defines the local-or-pull invariant that scheduling consumers rely on.

Tierless approximate match data remains eligible for source selection because
it can still identify a likely source. It does not confirm a CPU-tier prefix,
so it does not publish `ReusablePrefixTokens` for scheduling. A
[`context-length-aware`](../../../scheduling/scorer/contextlengthaware/README.md)
consumer leaves the total prompt length unchanged when no confirmed floor is
available.

## Parameters

- `prefixMatchInfoProducerName` (string, optional): Name of the prefix-cache producer instance that supplies `PrefixCacheMatchInfo`. Empty selects the default unnamed producer.
- `minCachedTokenDelta` (int, optional, default: `1`): Minimum cached-token advantage required to write the source. Must be `>= 1`. Higher values avoid transfers for prefixes that are cheap to recompute.
- `prefillProfileName` (string, optional, default: `prefill`): P/D disaggregation prefill profile containing the endpoint that computes the prefix. If the profile has no target, the primary profile target is used. See [Prefill profile name](#prefill-profile-name).
- `p2pConnectorPort` (int, optional, default: `7777`): `OffloadingConnector` P2P tier port of data parallel rank 0 on every endpoint; rank `r` listens on `p2pConnectorPort + r`, which must stay within 1-65535: an endpoint beyond that is skipped as a source, and a request to a prefill endpoint beyond it fails. Must match the P2P tier's configured port.
- `transport` (string, optional, default: `offloading`): How the source reaches the pod computing the prefix. `offloading` writes the outputs above for the vLLM `OffloadingConnector` P2P tier. `kvcr` writes a KVCR `kv.fetch` hint instead; see [KVCR transport](#kvcr-transport).
- `kvcrControlPort` (int, required with `transport: kvcr`): KVCR control port of data parallel rank 0 on every endpoint; rank `r` listens on `kvcrControlPort + r`.
- `costModel` (object, optional): When set, the source is written only when the pull's gain exceeds its cost, evaluated after the `minCachedTokenDelta` floor. Omitted, the fixed-delta decision applies. See [Cost model](#cost-model).
- `ttftByDecision` (bool, optional, default: `false`): Records the TTFT of each streaming request whose stream ends naturally in `llm_d_epp_p2p_source_request_ttft_seconds`, labelled with this producer's decision. Adds a response-body hook to the producer.

## Cost model

A pull saves the computing pod the prefill of `delta` tokens (the source's cached-token advantage) and costs the transfer of those tokens, a fixed setup cost, and a re-queue delay when the computing pod has a waiting queue: the pulled request is re-admitted through that queue after its blocks arrive. On a loaded pod a recompute also delays every running request's decode steps, while a pull delays only the pulled request; `fleetWeight` credits the pull with that freed prefill. The source is written when

```text
delta * (prefill - transfer) + fleetWeight * running(d) * delta * prefill
  > transferFixed + busy(s) * sourceWait + busy(d) * requeue
```

where `running(d)` is the number of requests running on the computing pod and `busy(x)` is true when pod `x` has at least `busyQueueThreshold` requests waiting. At `fleetWeight: 0` the decision optimizes the pulled request's own latency only.

The cost model also changes how the source is chosen. Instead of sampling within one block of the largest prefix weighted by queue depth, every source is ranked by the wait it adds (`sourceWait` if busy) plus the recompute the computing pod pays for the tokens it holds short of the best-cached source, `(maxCached - cached) * (prefill - transfer)`. Sources within one block's recompute of the minimum are sampled uniformly by request-ID hash. The source is chosen before the computing pod is known, so the ranking prices the shortfall without the `fleetWeight` credit; with `sourceWaitMs: 0` this does not change the ranking, and with a non-zero `sourceWaitMs` it undervalues cache for a loaded computing pod.

Parameters:

- `prefillMicrosecondsPerToken` (float, required): prefill cost per token on the computing pod.
- `transferMicrosecondsPerToken` (float, required): P2P transfer cost per token; must be below the prefill cost.
- `transferFixedMs` (float, optional, default `0`): fixed cost of a pull.
- `sourceWaitMs` (float, optional, default `0`): added when the source is busy. Pulls are served from the source's CPU tier, not its scheduler; a busy source measured no added pull latency on the fleets below.
- `requeueMs` (float, optional, default `0`): added when the computing pod has a waiting queue.
- `fleetWeight` (float, optional, default `0`): credit per running request on the computing pod, in units of the prefill time the pull saves.
- `busyQueueThreshold` (int, optional, default `1`): waiting-queue depth at or above which a pod counts as busy.

`prefillMicrosecondsPerToken` and `transferMicrosecondsPerToken` come from an idle pull-versus-recompute ladder (the `calibrate-min-cached-token-delta.sh` recipe in the llm-d guides prints both). `transferFixedMs` and `requeueMs` are larger under load than on an idle pod pair, because the engine re-queues a pulled request after its blocks arrive; measure them on a loaded fleet. Values measured for gpt-oss-120b, TP=1:

| Fleet | prefill | transfer | transferFixed | requeue | sourceWait |
| --- | --- | --- | --- | --- | --- |
| H200, pull over TCP | 33 | 16 | 400 | 400 | 0 |
| B200, pull over RDMA | 19 | 10 | 200 | 125 | 0 |

With `fleetWeight: 1`, a pod with a few requests running pulls any delta above the floor, and an idle pod pulls only deltas whose own prefill saving exceeds the fixed cost (about 23.5K tokens with the H200 values).

The [reusable prefix floor](#reusable-prefix-floor) is published in Produce from the source's cached prefix and `minCachedTokenDelta` alone; the cost model runs in PreRequest, after scheduling. A consumer such as `context-length-aware` can therefore plan prefill work around a pull the cost model declines: with the H200 values and an idle computing pod, every delta below about 23.5K tokens. When a consumer relies on the floor, set `minCachedTokenDelta` near the idle crossover, `transferFixedMs * 1000 / (prefillMicrosecondsPerToken - transferMicrosecondsPerToken)` tokens, so the floor and the decision agree on an idle pod.

```yaml
  - type: p2p-source-producer
    name: p2p-cache-source
    parameters:
      prefixMatchInfoProducerName: precise-cache
      minCachedTokenDelta: 256
      costModel:
        prefillMicrosecondsPerToken: 33
        transferMicrosecondsPerToken: 16
        transferFixedMs: 400
        requeueMs: 400
        fleetWeight: 1
```

## Configuration

Cache-aware scheduling requires confirmed per-tier data, so use a named
`precise-prefix-cache-producer`. Confirmed local-cache accounting excludes
speculative entries when `speculativeIndexing` is enabled. The token producer
must render tokens exactly as the serving model does. This example disables
speculative indexing and shows the producer-side configuration; the consuming
work classifier is documented under [P2P cache-aware prefill work](../../../scheduling/scorer/contextlengthaware/README.md#p2p-cache-aware-prefill-work).

```yaml
plugins:
  - type: token-producer
    parameters:
      modelName: model-name
      vllm:
        url: http://render:8000
  - type: endpoint-notification-source
  - type: precise-prefix-cache-producer
    name: precise-cache
    parameters:
      tokenProcessorConfig:
        blockSizeTokens: 64
      kvEventsConfig:
        topicFilter: "kv@"
        discoverPods: true
        podDiscoveryConfig:
          socketPort: 5557
      speculativeIndexing: false
  - type: p2p-source-producer
    name: p2p-cache-source
    parameters:
      prefixMatchInfoProducerName: precise-cache
      prefillProfileName: prefill
      minCachedTokenDelta: 1
dataLayer:
  sources:
    - pluginRef: endpoint-notification-source
      extractors:
        - pluginRef: precise-cache
```

### Prefill profile name

`prefillProfileName` must equal the name of the scheduling profile that picks the prefill
pod. Both names default to `prefill`, so a config that leaves them unset works. When you
rename the prefill profile, change `prefillProfileName` to the same name. If the names
differ, the plugin writes no `x-llm-d-kv-cache-prefill-p2p-host-port`, and every
disaggregated request fails: the sidecar `offloading` connector returns HTTP 500, and the
coordinator `kv-offloading` connector fails the request before decode is sent.

Behind the routing sidecar, the name comes from the `disagg-profile-handler`:

```yaml
plugins:
  - type: disagg-profile-handler
    parameters:
      profiles:
        prefill: prefill-h200       # the profile that picks the prefill pod
        decode: decode
  - type: p2p-source-producer
    parameters:
      prefillProfileName: prefill-h200   # must be the same name
schedulingProfiles:
  - name: prefill-h200
    plugins: [...]
  - name: decode
    plugins: [...]
```

Behind the coordinator, `header-profile-handler` runs the profile named by
`x-llm-d-epp-profile`, which the coordinator always sets to `prefill` on the prefill
request; with a single configured profile it runs that one whatever its name. Name the
prefill profile `prefill` and leave `prefillProfileName` unset:

```yaml
plugins:
  - type: header-profile-handler
  - type: p2p-source-producer       # prefillProfileName unset: prefill
schedulingProfiles:
  - name: prefill                   # equals prefillProfileName
    plugins: [...]
  - name: decode
    plugins: [...]
```

## KVCR transport

With `transport: kvcr`, the source goes to vLLM's KV Cache Runner (KVCR) tier
instead of the P2P tier. Source selection, the cost model, the reusable prefix
floor and the metrics are unchanged. In `PreRequest`, a kept source is written
into `GenerateRequest.kv_hints` of a `vllm.Inference` Generate or
GenerateStream request as one `kv.fetch` action whose
`source_control_endpoint` is `tcp://<source address>:<kvcrControlPort + rank>`,
replacing any `kv_hints` the request carries. The source's P2P body field and
header are not written. Other requests, such as OpenAI HTTP, get no hint,
publish no reusable prefix floor, and record the `unsupported-request` outcome.

On a disaggregated request, the hint names a source for the prefill pod.
`x-llm-d-kv-cache-prefill-p2p-host-port` is written as with `offloading`,
because the `offloading` P/D connector of the sidecar and the coordinator
reads it. The sidecar and the coordinator drop `kv_hints` from encode legs and
from the decode leg. A decode request that carries
`kv_transfer_params.remote_prefiller` or `do_remote_prefill: true` gets no
hint.

`kv.fetch` requires a non-empty `block_hashes` list. The plugin sends one
placeholder value and relies on KVCR's `opportunistic_query`, which vLLM's
KVCR tier enables by default: KVCR then treats every block of the request as
fetchable from the named source without reading the hashes. With
`opportunistic_query: false` the hint is accepted and fetches nothing.

The hint is field `kv_hints` of the `GenerateRequest` body. The client sends the
request without it, and the plugin adds it. For a source at `10.0.0.3` on data
parallel rank 1 with `kvcrControlPort: 17771`, a `GenerateStream` request
reaches vLLM as follows (shown as JSON with proto field names; `Generate` takes
the same message):

```json
{
  "request_id": "req-1",
  "model": "Qwen/Qwen3-8B",
  "text": "Summarize the policy document below.\n\n<document text>",
  "stopping": {"max_new_tokens": 128},
  "kv_hints": {
    "protocol_version": "0.1",
    "message_id": "<new UUID per request>",
    "actions": [
      {
        "action_id": "fetch-0",
        "action_type": "kv.fetch",
        "action_version": "1.0",
        "payload": {
          "source_control_endpoint": "tcp://10.0.0.3:17772",
          "block_hashes": [0]
        }
      }
    ]
  }
}
```

`opportunistic_query` is a setting of the vLLM `kvcr` secondary tier, not part
of the hint. To set it explicitly, add it to the tier in `--kv-transfer-config`:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 8000000000,
    "self_describing_kv_events": true,
    "secondary_tiers": [
      {
        "type": "kvcr",
        "router_capabilities": ["router_hint"],
        "control_host": "0.0.0.0",
        "control_ports": [17771],
        "control_advertise_host": "<pod IP>",
        "opportunistic_query": true
      }
    ]
  }
}
```

```yaml
  - type: p2p-source-producer
    parameters:
      prefixMatchInfoProducerName: precise-cache
      transport: kvcr
      kvcrControlPort: 17771
```

Requirements on every endpoint:

- vLLM serves `vllm.Inference` over gRPC, and `OffloadingConnector` has a `kvcr` secondary tier with `opportunistic_query` unset or `true`.
- The tier's `control_ports` are consecutive, starting at `kvcrControlPort` for local data parallel rank 0, and `control_advertise_host` is the pod address the EPP sees.
- The precise producer receives the tier's KV events. KVCR host memory reports as the `cpu` tier, the only tier a source is chosen from.

The EPP must parse the requests with `vllminference-parser`, listed under
`requestHandler.parsers`. Without that list, the EPP's default parsers send
gRPC requests to `passthrough-parser`, nothing is tokenized, and every request
records `no-best-match`.

## Deployment Requirements

The source results in a transfer only when the serving deployment can
serve and pull the named blocks:

- vLLM includes `OffloadingConnector` with a `p2p` secondary tier, and the routing sidecar runs the `offloading` connector (or `nixlv2` with `--enable-p2p-pull`).
- Every potential source uses `offload_prompt_only: false`. The default `true` omits decode-phase blocks from its offload tier.
- Peers use identical `--block-size` values. vLLM rejects a mismatch with `block_len mismatch`.
- Peers use the same `PYTHONHASHSEED`, so their block hashes agree.
- The precise producer receives KV events from every endpoint that may serve as a source and reports per-tier cache data.

For cache-aware prefill-work routing, follow the
[`context-length-aware` profile and work-range guidance](../../../scheduling/scorer/contextlengthaware/README.md#p2p-cache-aware-prefill-work).

## Metrics

- `llm_d_epp_p2p_source_decisions_total`: one decision per request, labelled by `endpoint_role` and `outcome`
  (`pulled`, `self`, `below-floor`, `declined`, and the cases that leave no decision to make).
- `llm_d_epp_p2p_source_predicted_pulled_tokens`: prompt tokens the selected source holds beyond
  the computing endpoint, per request sent to pull.
- `llm_d_epp_p2p_source_unpulled_tokens`: the same gap, per request the producer did not send to
  pull, labelled by `outcome` (`declined`, `below-floor`).
- `llm_d_epp_p2p_source_request_ttft_seconds`: TTFT of streaming requests by decision and cost-model
  margin, with `ttftByDecision`.

See [P2P KV-cache source](../../../../../../../docs/metrics.md#p2p-kv-cache-source) for every
outcome, how `endpoint_role` is assigned under a renamed prefill profile, and the vLLM metrics that
report the pulls served.

## Related Documentation

- [Context Length Aware Scorer](../../../scheduling/scorer/contextlengthaware/README.md)
- [Approximate Prefix Cache Producer](../approximateprefix/README.md)
- [Precise Prefix Cache Producer](../preciseprefixcache/README.md)
