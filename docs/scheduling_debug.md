# Scheduling Decisions Debug Endpoint

The router exposes the most recent scheduling decisions on the metrics/admin server:

```text
/debug/scheduling/recent
```

The endpoint answers operational questions that metrics and sampled traces only partly cover: why a request was routed to a specific pod, which filter removed candidates, and which scorer's scores separated the picked endpoint from the rest.

The endpoint is disabled by default. Enable it by starting the EPP with:

```text
--scheduling-decisions-history=100
```

The value is the number of recent decisions retained in a fixed-capacity ring buffer. `--scheduling-decisions-top-endpoints` (default 5) caps how many of a profile's highest-scoring endpoints are recorded per decision. The recorded model string is truncated to 1024 bytes. These bounds keep the retained state bounded regardless of traffic.

The endpoint is registered on the same metrics/admin server as the metrics endpoint and pprof handlers. Operators should use the existing metrics/admin server exposure and authentication controls to restrict access. When metrics endpoint authentication is enabled, the metrics/admin server authentication and authorization filters apply to this endpoint.

This endpoint is available in the EPP controller-manager server path. The standalone file-discovery mode does not serve it: its metrics server applies no metrics-endpoint authentication, so the EPP refuses to start when `--scheduling-decisions-history` is set on that path.

## Response format

```json
{
  "timestamp": "2026-01-01T00:00:00Z",
  "decisions": [
    {
      "timestamp": "2026-01-01T00:00:01Z",
      "requestID": "b0c1b5a4-...",
      "model": "my-model",
      "profiles": [
        {
          "profile": "default",
          "candidatesBeforeFilters": 12,
          "filters": [
            {
              "type": "decode-filter",
              "name": "decode-filter",
              "before": 12,
              "after": 4
            }
          ],
          "scorers": [
            {
              "type": "kv-cache-utilization-scorer",
              "name": "kv-cache-utilization-scorer",
              "weight": 1.0
            }
          ],
          "topEndpoints": [
            {
              "endpoint": "default/pod1",
              "weightedScore": 0.83,
              "scorerScores": {
                "kv-cache-utilization-scorer/kv-cache-utilization-scorer": 0.83
              }
            }
          ],
          "targetEndpoints": ["default/pod1"]
        }
      ],
      "targetEndpoint": "default/pod1"
    }
  ]
}
```

Entries are returned newest first. Each decision covers one scheduling attempt and holds:

- `requestID` and `model`: the request identity. No request payloads or headers are recorded.
- `profiles`: one entry per profile run. A profile that the profile handler runs in several iterations produces one entry per run.
  - `filters`: each filter's candidate count before and after it ran.
  - `scorers`: each scorer's identity and configured weight.
  - `topEndpoints`: the highest-scoring endpoints by weighted score, each with the raw score every scorer assigned it. `scorerScores` keys are `<name>/<type>`.
  - `targetEndpoints`: the endpoints the profile's picker selected.
  - `error`: the profile run's error, present only when the run failed.
- `targetEndpoint`: the primary profile's picked endpoint.
- `failureReason`: the reason the attempt failed, matching a `reason` value of `llm_d_epp_scheduler_failures_total`; empty on success.

A decision with no `profiles` entries and a `failureReason` of `no_profile_selected` means the profile handler picked no profile to run. A profile entry whose last filter shows `after: 0` identifies the filter that drained the candidate set for a `failureReason` of `all_filtered`.
