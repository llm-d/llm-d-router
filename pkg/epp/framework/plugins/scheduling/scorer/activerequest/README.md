# Active Request Scorer

**Type:** `active-request-scorer`

Scores pods based on the number of active requests being served per pod. It consumes request counts produced by the `inflight-load-producer`.

Pods at or below `idleThreshold` active requests receive the maximum score (`1.0`). Busier pods are scored proportionally in the range `[0, maxBusyScore]`, with the most-loaded pod scoring lowest.

**Parameters:**
- `idleThreshold` (int, optional, default: `0`): Request count at or below which a pod is considered idle and scores `1.0`.
- `maxBusyScore` (float, optional, default: `1.0`): Maximum score assigned when a pod is busy.
- `requestTimeout` (duration string, optional): Deprecated and ignored. Accepted only for backward compatibility; request lifecycle is tracked by `inflight-load-producer`.
- `scoreAtPick` (bool, optional, default: `false`): Score at pick time. The scheduler profile runs the scorer after its other scorers, inside a critical section that also covers the picker and the reservation of the picked endpoint, so each request sees the requests reserved by every pick before it. A burst of concurrently scheduled requests is then spread as if its requests had been scheduled one at a time.

**Configuration Example:**
```yaml
plugins:
  - type: active-request-scorer
    name: active-tracker
    parameters:
      idleThreshold: 2
      maxBusyScore: 0.5
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: active-tracker
        weight: 5
```
