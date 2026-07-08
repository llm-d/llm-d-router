# Diffusion Cost Scorer

**Type:** `diffusion-cost-scorer`

Scores endpoints by the outstanding declared cost of their in-flight image generation requests, consumed from the `diffusion-load-producer`. Where active-request scoring treats every request as equal work, this scorer weighs each request by its declared diffusion cost (inference steps x output resolution x image count), so two queued low-step thumbnails do not count the same as one queued high-step full-size render.

Endpoints with no outstanding cost receive the maximum score (`1.0`). Loaded endpoints are scored proportionally, with the most-loaded endpoint scoring `0`.

Requests that are not image generation requests contribute no cost; on a mixed-traffic pool, pair this scorer with `active-request-scorer` or `load-aware-scorer` so non-diffusion load is still visible.

**Parameters:**
- `diffusionLoadProducerName` (string, optional): Instance name of the `diffusion-load-producer` whose attribute to read. Defaults to the producer registered under its type name.

**Configuration Example:**
```yaml
plugins:
  - type: diffusion-cost-scorer
    name: diffusionCost
schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: diffusionCost
        weight: 5
```
