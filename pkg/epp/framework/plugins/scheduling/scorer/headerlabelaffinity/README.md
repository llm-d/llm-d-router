# Header Label Affinity Scorer

**Type:** `header-label-affinity-scorer`
**Interfaces:** `scheduling.Scorer`, `requestcontrol.ResponseHeaderProcessor`

Adds soft affinity for endpoints whose configured label equals a request
header. A matching endpoint receives a score of `1`; every other endpoint
receives `0` and remains eligible.

Use a separate plugin instance for each header-to-label mapping. This allows
each preference to have its own scheduling-profile weight.

## Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| `headerName` | string | Yes | Request header containing the preferred label value. |
| `labelKey` | string | Yes | Endpoint label compared with the request header. |
| `stampResponseHeader` | boolean | No | Copies the selected endpoint's label into `headerName` on the response. Defaults to `true`; set it to `false` when the header is input-only. |

## Configuration

```yaml
plugins:
- type: header-label-affinity-scorer
  name: slice-affinity
  parameters:
    headerName: x-disagg-slice
    labelKey: disaggregatedset.x-k8s.io/slice
- type: header-label-affinity-scorer
  name: zone-affinity
  parameters:
    headerName: x-preferred-zone
    labelKey: topology.kubernetes.io/zone
    stampResponseHeader: false
- type: weighted-random-picker
  name: picker

schedulingProfiles:
- name: decode
  plugins:
  - pluginRef: slice-affinity
    weight: 3
  - pluginRef: zone-affinity
    weight: 1
  - pluginRef: picker
```

The default lifecycle is:

1. With no request header, every endpoint gets affinity score `0`, so the other
   configured scorers and picker choose an endpoint.
2. The scorer stamps that endpoint's actual label into the response header.
3. A coordinator can copy the response header into a later request, where it
   becomes a soft preference.

When a request already contains the header, the response is still stamped with
the endpoint that was actually selected. Because affinity is soft, other scores
can make that label differ from the requested preference. Set
`stampResponseHeader: false` only when the header is an input hint that must not
be returned.

## DisaggregatedSet Slice Affinity

A `DisaggregatedSet` can replicate its complete role topology into independent
slices. For example, this creates two copies of the prefill and decode
topology:

```yaml
apiVersion: disaggregatedset.x-k8s.io/v1
kind: DisaggregatedSet
metadata:
  name: my-set
spec:
  slices: 2
  roles:
  - name: prefill
    spec:
      replicas: 2
      # leaderWorkerTemplate omitted
  - name: decode
    spec:
      replicas: 10
      # leaderWorkerTemplate omitted
```

The controller labels every Pod in each topology copy with
`disaggregatedset.x-k8s.io/slice`. In this example the values are `0` and `1`.
When each slice is placed within one NVL72 domain, preferring the slice selected
for an earlier role can avoid a cross-domain KV-cache transfer.

The affinity scorer stamps the selected prefill Pod's slice into
`x-disagg-slice`, then uses that request header to prefer the same slice for
decode:

```yaml
plugins:
- type: disaggregatedset-rollout-screener
  name: rollout-screener
  parameters:
    scope:
      labelSelector: disaggregatedset.x-k8s.io/name=my-set
    revisionGating:
      revisionHeaderName: x-llm-d-disagg-revision
      revisionLabelKey: disaggregatedset.x-k8s.io/revision
      roleLabelKey: disaggregatedset.x-k8s.io/role
      mode: max-role
      requiredRoles: [prefill, decode]
- type: header-label-affinity-scorer
  name: slice-affinity
  parameters:
    headerName: x-disagg-slice
    labelKey: disaggregatedset.x-k8s.io/slice
- type: weighted-random-picker
  name: picker

schedulingProfiles:
- name: decode
  plugins:
  - pluginRef: slice-affinity
    weight: 3
  - pluginRef: picker
```

The component coordinating the roles must copy `x-disagg-slice` from the
prefill response into the decode request. The weight determines how strongly
same-slice locality competes with other decode scorers; it does not make the
slice a hard requirement.

## Operational Notes

- A missing request header contributes zero to every endpoint.
- An unknown header value contributes zero to every endpoint.
- Missing endpoint labels receive zero.
- The scheduling profile multiplies the score by the plugin weight and adds it
  to the other weighted scorer results.
