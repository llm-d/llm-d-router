# Served Model Filter Plugin

**Type:** `served-model-filter`

Keeps only the endpoints whose model server lists the requested model.
Enable this Alpha plugin with `--allow-experimental-plugins=true`.

## What it does

vLLM answers `GET /v1/models` with its base model and every LoRA adapter registered with it,
and marks each adapter with a `parent`. Registered means servable: an adapter evicted from the
GPU or CPU cache stays listed and is reloaded on its next request. The `models-data-source`
polls that endpoint and the `models-data-extractor` stores the answer per endpoint. For a
request whose target model is `X`, this filter keeps an endpoint if its stored list contains
`X`.

The [model server protocol](../../../../../../../docs/plugin-metric-protocol.md#lora-adapter-serving)
assumes every endpoint can serve every valid adapter. A pool where a controller registers
adapters on some endpoints only, through vLLM's run-time management API, does not meet it.
There, requests for an adapter are routed by load and prefix cache alone and fail with
`404 model does not exist` on endpoints that lack the adapter, and this filter narrows the set
to the endpoints that list it. The `lora-affinity-scorer` reads `vllm:lora_requests_info`,
which lists adapters with requests in flight, and scores endpoints without excluding any.

## Behavior

| Situation | Result |
|---|---|
| Endpoint lists the target model | Kept |
| Endpoint lists other models only | Dropped |
| Endpoint has no stored list, and some endpoint lists the target as an adapter | Dropped |
| Endpoint has no stored list, otherwise | Kept; dropped with `onMissing: Fail` |
| No endpoint is kept | Empty result, and the request fails with 503; or the unfiltered set with `fallbackOnEmpty: true` |
| Request has no target model | Unfiltered set |

An endpoint has no stored list until its first successful poll, which runs on its first
collection tick, 50ms after the endpoint is added by default. On each tick the sources poll
an endpoint one after another, and ticks that arrive while a round is still running are
skipped. `interval` is counted in ticks, so a slow scrape by another source delays the first
poll and stretches the interval. A retry after a failed poll waits at least one `interval`.

Every endpoint serving a base model lists it, so keeping endpoints without a list for
base-model requests spreads load across endpoints that have not reported yet. For an adapter
that a listed endpoint serves, they are dropped because they are unlikely to have it registered.
In `endpoint-attribute-filter`, `onMissing: Pass` keeps endpoints without the attribute in
every case.

With `onMissing: Fail`, a data-source failure that affects every endpoint (for example a
model server that requires an API key on `/v1/models`) fails every request, including
base-model requests.

A failed poll leaves the previous list in place, and an endpoint that becomes NotReady or is
deleted loses its attributes. A model server restart that the endpoint reconciler did not
observe keeps the old list until the next successful poll.

The target model is empty for the `passthrough-parser`, `sglanghttp-parser`,
`vertexai-parser` and `vllmgrpc-parser` unless the `x-llm-d-model-name-rewrite` header sets
it, so the filter does not narrow those requests.

## Placement

The filter judges each endpoint against the other candidates, so its position in a profile
matters:

- **After role and label filters.** In a decode profile, a prefill endpoint that lists the
  adapter would otherwise cause decode endpoints without a list to be dropped, and the role
  filter would then drop the prefill endpoint.
- **Before session and prefix-cache affinity filters.** An affinity filter placed first can
  keep only an endpoint that lacks the adapter.

In disaggregated serving, prefill and decode endpoints both need the adapter, and every
profile sees the same target model. Add the filter to the prefill and the decode profile. An
empty prefill profile fails the request; there is no decode-only fallback.

## Runtime LoRA resolvers

vLLM can load an adapter on its first request through a runtime LoRA resolver. Before that
request no endpoint lists the adapter, so once every endpoint has a stored list the request
fails with 503 unless `fallbackOnEmpty: true` is set. The same 503 answers a request for a
model that does not exist, where vLLM would return 404. Once an endpoint lists the adapter,
every later request for it goes to the endpoints that list it, so the resolver does not load
it anywhere else. Registering adapters on more endpoints is left to whatever manages them.

## Inputs consumed

- `/v1/models` attribute from `models-data-extractor` (`attrmodels.ModelDataCollection`),
  declared as a required dependency: a configuration without `models-data-extractor` fails
  at start-up. An extractor that is declared but cannot be bound to a `models-data-source`
  only logs `datalayer: skipping unresolved dependency`. Every endpoint then lacks a list, and
  with the default `onMissing: Pass` the filter keeps them all.

## Configuration

| Parameter | Default | Meaning |
|---|---|---|
| `onMissing` | `Pass` | `Fail` drops every endpoint without a stored list |
| `fallbackOnEmpty` | `false` | Return the unfiltered set when no endpoint is kept |

**Configuration example:**

```yaml
plugins:
  - type: models-data-source
    parameters:
      interval: 5s
  - type: models-data-extractor
  - type: served-model-filter
  - type: queue-scorer
  - type: max-score-picker
dataLayer:
  sources:
    - pluginRef: models-data-source
      extractors:
        - pluginRef: models-data-extractor
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: served-model-filter
      - pluginRef: queue-scorer
      - pluginRef: max-score-picker
```

## Metrics

`llm_d_epp_served_model_filter_decisions_total{plugin_name, outcome}` counts one outcome per
call. A call is one profile run, so a request that runs through a prefill and a decode profile
records two decisions under the same `plugin_name` unless each profile has its own instance.

| Outcome | Meaning |
|---|---|
| `listed` | At least one endpoint lists the target model |
| `unlisted` | No endpoint lists the target model, and the endpoints without a list were kept |
| `fallback` | No endpoint was kept, and `fallbackOnEmpty` returned the unfiltered set |
| `empty` | No endpoint was kept, so the request fails |
| `not_applicable` | The request has no target model |

An instance records either `fallback` or `empty`, depending on `fallbackOnEmpty`.

`llm_d_epp_served_model_filter_candidates_total{plugin_name}` and
`llm_d_epp_served_model_filter_unlisted_candidates_total{plugin_name}` count the candidate
endpoints the filter evaluated and those without a stored list. The ratio of their rates is
the share of candidates without a list, weighted by filter calls. It has no value while the
filter receives no traffic.

The default `plugin_name` is the same in every EPP, so group by `job` as well as `namespace`.
Both signals below are tickets. Under `onMissing: Pass` a failing models source shows up as
`unlisted` rather than `empty`, and requests for a model that exists nowhere, such as a client
typo, also count as `empty`, so neither makes a good page. Page on the request-level failure
ratio of `llm_d_epp_scheduler_attempts_total{status="failure"}`, which counts each request once.

Requests failing because no endpoint lists their model, with a traffic floor, `for: 15m`:

```promql
(
  sum by (namespace, job, plugin_name) (rate(llm_d_epp_served_model_filter_decisions_total{outcome="empty"}[15m]))
  /
  sum by (namespace, job, plugin_name) (rate(llm_d_epp_served_model_filter_decisions_total{outcome!="not_applicable"}[15m]))
  > 0.05
)
and
sum by (namespace, job, plugin_name) (rate(llm_d_epp_served_model_filter_decisions_total{outcome!="not_applicable"}[15m])) > 0.1
```

Endpoints without a list, meaning polls are failing or the extractor is unbound, `for: 15m`:

```promql
sum by (namespace, job, plugin_name) (rate(llm_d_epp_served_model_filter_unlisted_candidates_total[15m]))
/
sum by (namespace, job, plugin_name) (rate(llm_d_epp_served_model_filter_candidates_total[15m]))
> 0.1
```

## Troubleshooting

**503s for one model** (`empty` rising). Find the model with
`topk(5, sum by (target_model_name) (rate(llm_d_epp_request_error_total{error_code="ServiceUnavailable"}[10m])))`
and compare the name with `GET /v1/models` on a pod. Fix the client or the model rewrite if
the name is wrong. If the adapter should exist, check whatever registers it. If adapters load
on their first request, set `fallbackOnEmpty: true`.

**The filter does not narrow** (unlisted share near 1). Check
`llm_d_epp_datalayer_poll_errors_total{source_type="models-data-source"}`. Without poll errors,
look for `datalayer: skipping unresolved dependency` in the EPP log and list the extractor
under the `models-data-source` in `dataLayer.sources`.

**Polls to `/v1/models` fail** (poll errors rising). Request `<scheme>://<pod-ip>:<port>/v1/models`
from the EPP pod's network, for example with `kubectl debug --target=<epp container>`. A
connection or TLS error points at a NetworkPolicy or the CA bundle, a 401 at an API key. The
CA bundle is read at start-up, and a restart drops every stored list, so restart the EPP only
when polls will succeed afterwards.

**404s for an adapter despite the filter.** `not_applicable` means the request has no target
model, and `fallback` means it went to the unfiltered set. `listed` right after an adapter was
unregistered means the stored list was up to one `interval` old. Lower `interval`.

**One adapter overloads its endpoints.** Flow-control saturation is pool-wide and does not
shed load from the few endpoints that list an adapter. Register the adapter on more endpoints.
