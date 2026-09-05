# DisaggregatedSet Rollout Screener

**Type:** `disaggregatedset-rollout-screener`
**Interfaces:** `requestcontrol.Screener`, `requestcontrol.ResponseHeaderProcessor`

This plugin prevents incompatible prefill and decode Pods from serving the same
request during a `DisaggregatedSet` rollout. It runs before every scheduling
profile, so the compatibility constraint cannot be undone by a later filter,
scorer, or picker.

## Why Revision Screening Is Needed

A `DisaggregatedSet` revision represents one complete version of the whole
disaggregated deployment, including every role. If the configuration of any
role changes, the controller creates a new revision for the entire
`DisaggregatedSet`. All prefill and decode Pods created for that version receive
the same `disaggregatedset.x-k8s.io/revision` label, while Pods from the previous
version retain their old revision label.

The router uses this label as a compatibility boundary: a prefill Pod and a
decode Pod may serve the same request only when their revision labels are
equal. For example, it may pair a prefill Pod labeled `rev-A` with a decode Pod
labeled `rev-A`, but never a prefill Pod labeled `rev-A` with a decode Pod
labeled `rev-B`.

Pairing Pods from different revisions can be troublesome:

- The revisions can use incompatible KV-cache formats.
- Switching libraries or inference-engine versions can change the KV-cache
  transfer protocol without backward compatibility.
- An unreliable old revision can produce corrupted KV-cache state that should
  not reach the new revision.
- A rollout can move Pods to a different driver through a `nodeSelector` or
  toleration. Old and new drivers can be incompatible with the KV-cache
  transfer mechanism.

Rolling out all roles at exactly the same percentage is not always possible.
Replica counts are integers, and each role is constrained by its own surge and
unavailable limits. For example, consider a deployment whose stable shape is
2 prefill Pods and 10 decode Pods. An intermediate state can be:

```text
                 prefill   decode
old revision A      2         9
new revision B      1         1
```

Selecting a revision from the prefill pool alone would send about 67% of
requests to A and 33% to B. However, B has only 1 of the 10 decode Pods and can
represent only about 10% of decode capacity. That can overload B's decode side
while leaving A's decode capacity unused.

The
[DisaggregatedSet rollout KEP](https://github.com/kubernetes-sigs/lws/tree/main/keps/766-DisaggregatedSet)
explains how the controller keeps role progress as close as integer replica
counts permit. To inspect a concrete rollout, run the planner from the root of a
[`kubernetes-sigs/lws`](https://github.com/kubernetes-sigs/lws) checkout. This
example uses a 2P:10D deployment with `maxUnavailable: 2` for both roles:

```bash
cd "$(go env GOPATH)/src/github.com/kubernetes-sigs/lws"
go run ./hack/plan-steps \
  --source '{"prefill": 2, "decode": 10}' \
  --target '{"prefill": 2, "decode": 10}' \
  --surge '{"prefill": 0, "decode": 0}' \
  --unavailable '{"prefill": 2, "decode": 2}'
```

```text
Roles: [decode prefill]
Source: decode=10, prefill=2
Target: decode=10, prefill=2
Config: decode(surge=0, unavailable=2), prefill(surge=0, unavailable=2)

Step  Old decode  Old prefill  New decode  New prefill  Total  Action
----  ----------  -----------  ----------  -----------  -----  -----------------------------
0     10          2            0           0            12     initial
1     8           2            0           0            10     old decode -2
2     8           2            2           0            12     new decode +2
3     6           1            2           0            9      old decode -2, old prefill -1
4     6           1            4           1            12     new decode +2, new prefill +1
5     4           1            4           1            10     old decode -2
6     4           1            6           1            12     new decode +2
7     2           1            6           1            10     old decode -2
8     2           1            8           1            12     new decode +2
9     0           0            8           1            9      old decode -2, old prefill -1
10    0           0            10          2            12     new decode +2, new prefill +1
```

At steps 2 and 3, the new revision has no prefill Pod, so it cannot serve a
request. Once both revisions are covered, decode is the globally largest role.
`max-role` therefore produces these old/new revision shares:

- Step 4: 60% old, 40% new.
- Step 6: 40% old, 60% new.
- Step 8: 20% old, 80% new.

## Request Lifecycle

For every Pod notification, the plugin caches the Ready Pod count by revision
and role. For a request without a strict revision header, it then:

1. Removes every revision that has no Ready Pod for any required role, because
   a revision missing one role cannot serve the request.
2. Computes a weight for each remaining revision.
3. Randomly chooses one candidate revision using those weights.
4. When the request carries `x-llm-d-revision-decision-id`, or falls back to
   `x-request-id`, atomically stores or reads the revision decision through the
   configured `CrossReplicaSyncer`. Without one, it stores the decision in the
   local EPP process.
5. Exposes only the resulting revision's endpoints to all scheduling profiles.
6. Stamps the selected endpoint's revision into the configured response header.

```text
Ready Pod counts
       |
       v
remove incomplete revisions -> choose one covered revision
       |                               |
       +-------------------------------+
                                       v
                        filters, scorers, and picker
                                       |
                                       v
                         stamp the selected revision
```

When a strict revision header is already present, the plugin does not make a
new weighted choice. It checks that the requested revision has all required
roles and keeps only endpoints with that revision.

### Separate Prefill and Decode EPPs (P/D)

Prefill first chooses a covered revision and stamps it into the
`x-llm-d-disagg-revision` response header. The coordinator must copy that header into
the decode request. Decode treats it as a strict constraint and never calls
`GetOrSet`, so the prefill and decode EPPs do not need to share a
`CrossReplicaSyncer`:

```text
prefill request -> choose revision B -> x-llm-d-disagg-revision: B
                                             |
                                             v
decode request  -> strict revision B -> no GetOrSet
```

A decode request without the forwarded `x-llm-d-disagg-revision` does not implement
the supported P/D protocol. Do not rely on `GetOrSet` to coordinate separate
prefill and decode EPPs.

### Parallel Encode Requests (E/P/D)

Parallel encode requests cannot wait for an earlier response to provide
`x-llm-d-disagg-revision`. The coordinator gives them the same
`x-llm-d-revision-decision-id`, and atomic `GetOrSet` makes the first proposed
covered revision authoritative. Requests that can reach different EPP replicas
must share a `CrossReplicaSyncer`; a single EPP process can use the local
fallback. As soon as a phase response supplies `x-llm-d-disagg-revision`, the
coordinator forwards that header to later requests, which use strict filtering
instead of `GetOrSet`.

### One EPP

The Screener runs once before the disaggregated scheduling profiles. Choosing
one revision up front gives the decode and prefill profiles the same restricted
candidate set, so they cannot independently select different revisions.

## Revision Gating Modes

### `max-role`

The plugin first totals each required role across all covered revisions and
selects the globally largest role. It then weights every revision using that
same role:

```text
dominantRole = argmax(role) sum(Ready Pods for role across covered revisions)
weight(revision) = Ready Pods for dominantRole in revision
```

For the 2P:10D example:

```text
global totals: prefill = 3, decode = 10
dominant role: decode
A: 9 decode Pods
B: 1 decode Pod
traffic: A 90%, B 10%
```

The dominant role is selected once for the whole distribution. It cannot be
prefill for one revision and decode for another. If role totals are equal, the
first role in `revisionGating.requiredRoles` wins the tie.

This mode is useful for a stable, intentionally asymmetric role ratio such as
2P:10D or 10P:2D. It assumes that the more numerous role is a reasonable proxy
for the deployment's traffic-limiting capacity. It is less reliable when the
deployment changes between prefill-heavy and decode-heavy shapes or when the
role ratios differ substantially between revisions.

### `sum`

The weight of a revision is the total number of Ready Pods across the required
roles:

```text
weight(revision) = sum(Ready Pods for every required role)
```

For the same example:

```text
A: 2 + 9 = 11
B: 1 + 1 = 2
traffic: A 84.6%, B 15.4%
```

`sum` uses progress from every role and is a more general heuristic when role
ratios can change because of scaling, readiness changes, or a topology change.
When every revision has the same P:D ratio, `sum` and `max-role` produce the
same traffic percentages.

Neither mode is an exact capacity model. Exact weighting would require the
relative request capacity of a prefill Pod and a decode Pod, not only their
counts.

### `disabled`

This mode disables revision coverage checks and weighted revision selection.
Strict revision selection and response-header stamping remain enabled:

- With no revision header, all located candidates continue to filters,
  scorers, and the picker.
- The selected endpoint's revision is still stamped on the response.
- With a revision header, only matching endpoints remain. No match fails
  closed.

This mode does not make a revision decision. It cannot keep separate EPPs, or
the profiles of a single EPP, on one revision.

## Revision Header

The revision header is always strict. When a request supplies it, the Screener
keeps only endpoints whose revision label has the requested value. With an
active gating mode, that revision must also have a Ready Pod for every required
role. No match fails closed.

The Screener stamps the selected endpoint's revision into the same response
header. Other label affinities, such as slice affinity, belong in a
[`header-label-affinity-scorer`](../../../scheduling/scorer/headerlabelaffinity/README.md).

## Configuration

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: EndpointPickerConfig
plugins:
- type: disaggregatedset-rollout-screener
  name: rollout-screener
  parameters:
    scope:
      labelSelector: "disaggregatedset.x-k8s.io/name=my-set"
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

The Screener is discovered from the top-level `plugins` list and runs once per
request. Do not add it to a scheduling profile.

## Parameters

| Name | Type | Required | Default | Description |
|---|---|---|---|---|
| `scope.labelSelector` | string | Yes | | Selects the Pods observed for cross-role revision coverage. |
| `revisionGating` | object | Yes | | Revision screening configuration. |
| `revisionGating.mode` | string | Yes | | `sum`, `max-role`, or `disabled`. |
| `revisionGating.requiredRoles` | array | Yes for `sum` and `max-role` | | Roles that must each have a Ready Pod for a revision to receive traffic. List order breaks a `max-role` tie. |
| `revisionGating.revisionHeaderName` | string | No | `x-llm-d-disagg-revision` | Request and response header carrying the rollout revision. A supplied value is a strict constraint. |
| `revisionGating.revisionLabelKey` | string | No | `disaggregatedset.x-k8s.io/revision` | Label identifying a rollout revision. |
| `revisionGating.roleLabelKey` | string | No | `disaggregatedset.x-k8s.io/role` | Label identifying a Pod role. |

## DisaggregatedSet Slice Affinity

A `DisaggregatedSet` slice groups cooperating role replicas through the
`disaggregatedset.x-k8s.io/slice` label. A slice can represent endpoints placed
within one NVL72 NVLink domain. Preferring the same slice can avoid a slower
cross-domain KV-cache transfer while retaining fallback capacity when that
slice is unavailable or overloaded.

Strict revision selection is supported in P/D by forwarding
`x-llm-d-disagg-revision`. E/P/D uses the shared decision ID only for parallel
requests that start before that header exists. Slice selection is a soft
preference, so KV-cache and load-aware scoring can select another slice when it
is a better candidate. The affinity scorer returns the actually selected slice
to the coordinator by default.

The scorer weight represents how preferable the same NVL72 domain is relative
to the other scorers in the profile. Benchmark same-slice and cross-slice KV
transfers with representative traffic. Choose a weight that normally avoids a
cross-domain transfer but still lets health and load scorers select another
endpoint when the same-slice endpoint is overloaded.

A matching endpoint receives the full configured weight and a non-matching
endpoint receives zero. For example, assume a load scorer has weight `5`. If
the same-slice endpoint receives a load score of `0.4` and the cross-slice
endpoint receives `0.9`, their weighted load scores are `2.0` and `4.5`. The
cross-slice endpoint has a `2.5` advantage, so the slice weight must be greater
than `2.5` for the same-slice endpoint to win in that example. Production
values depend on the other configured scorers and their runtime scores.

## Fail-Closed Behavior

With `sum` or `max-role`, a revision is ineligible until every required role
has at least one Ready Pod. If no revision survives, request control returns
HTTP 503.

After EPP startup or restart, the in-memory distribution is empty until Pod
notifications deliver the initial snapshot. During that interval, `sum` and
`max-role` return HTTP 503 rather than route without cross-role coverage
information. A follow-up change is required to make receipt of the initial Pod
snapshot a condition of EPP readiness.

A strict header with no matching endpoint also returns HTTP 503. The plugin
never silently substitutes another revision or crosses revisions.

## Metrics

- `llm_d_epp_disaggregatedset_strict_revision_no_match_total`: strict revision
  selections that matched no endpoint and failed closed.
- `llm_d_epp_disaggregatedset_revision_gating_share`: current weighted share
  from `0` to `1` for each observed revision. Incomplete revisions report `0`.

The share gauge removes a revision's series when that revision disappears from
the observed Pod set.
