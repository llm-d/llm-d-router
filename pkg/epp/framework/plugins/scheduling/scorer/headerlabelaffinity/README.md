# Header Label Affinity Scorer

**Type:** `header-label-affinity-scorer`
**Interface:** `scheduling.Scorer`

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

The scorer does not write response headers. Protocols that return the selected
label to a later request must configure response-header stamping separately.

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

The rollout Screener stamps the selected prefill Pod's slice into
`x-disagg-slice`. The generic scorer repeats the mapping so the decode profile
prefers Pods from that slice:

```yaml
plugins:
- type: disaggregatedset-rollout-screener
  name: rollout-screener
  parameters:
    scope:
      labelSelector: disaggregatedset.x-k8s.io/name=my-set
    headerSelectors:
    - name: revision
      headerName: x-disagg-revision
      labelKey: disaggregatedset.x-k8s.io/revision
      mode: strict
    - name: slice
      headerName: x-disagg-slice
      labelKey: disaggregatedset.x-k8s.io/slice
      mode: prefer
    revisionGating:
      mode: max-role
      requireRoles:
        values: [prefill, decode]
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
