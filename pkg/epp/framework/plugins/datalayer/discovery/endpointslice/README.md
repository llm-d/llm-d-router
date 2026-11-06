# EndpointSlice Discovery Plugin

**Type:** `endpointslice-discovery`
**Interface:** `EndpointDiscovery`

Builds the endpoint datastore from Kubernetes EndpointSlices instead of Pods.

## What It Does

Default Kubernetes discovery reads `pod.status.podIP` of the Pods an
`InferencePool` selects. That is not usable when the model servers are not
Pods with routable Pod IPs, for example servers running outside the cluster,
or `hostNetwork` Pods that all report one node IP while the reachable
addresses live elsewhere. A controller that knows the real addresses can
publish them as EndpointSlices of a selector-less Service; this plugin reads
those slices.

## How It Works

- **Watch.** On `Start`, the plugin creates its own client and an informer on
  EndpointSlices in `namespace` that match `selector`. The discovery path runs
  without a controller manager, so the plugin does not share a manager cache.
- **Reconcile.** Every informer event triggers a level-triggered reconcile on
  the `Start` goroutine: the endpoint set is recomputed from the informer
  cache and the difference from the previous reconcile is applied with
  `notifier.Upsert` and `notifier.Delete`. An endpoint whose address, port,
  labels and attributes are unchanged is not upserted again.
- **Endpoints.** Each slice endpoint becomes one EPP endpoint, using its first
  address (the only one the EndpointSlice API defines as meaningful) and the
  slice port selected by `portName`. The endpoint ID is
  `<slice namespace>/<service>.<hostname>`, falling back to the address when
  the endpoint has no hostname, and `<service>` falls back to the slice name.
  An endpoint listed by two slices of one Service is reported once.
- **Readiness.** Endpoints whose `ready` condition is false, or whose
  `terminating` condition is true, are not reported. An unset `ready`
  condition counts as ready.
- **Labels.** Slice labels are copied onto the endpoint unchanged, so the
  controller that writes the slices sets the keys EPP plugins read, for
  example `llm-d.ai/role` for the role filters and `llm-d.ai/engine-type` for
  the metrics extractor.
- **Ready channel.** `Ready()` is closed after the first reconcile of the
  synced cache.

## Inputs Consumed

`discovery.k8s.io/v1` EndpointSlices with `addressType` `IPv4` or `IPv6`.
Slices of other address types are skipped.

## Configuration

**Location:** `dataLayer.discovery.endpoints.pluginRef` referencing a plugin entry of
type `endpointslice-discovery` in `plugins`.
**Enabled by default:** No.
**Stability:** Alpha (requires `--allow-experimental-plugins`).

### Parameters

| Name | Type | Required | Default | Description |
|------|------|----------|---------|-------------|
| `namespace` | `string` | yes | -- | Namespace of the EndpointSlices. |
| `selector` | `string` | no | all slices | Label selector, for example `kubernetes.io/service-name=model-server`. |
| `portName` | `string` | no | -- | Slice port to route to. When empty, a slice must declare exactly one port. |
| `kubeconfig` | `string` | no | in-cluster | Kubeconfig path. When empty, the in-cluster config is used, then `$KUBECONFIG`. |

### RBAC

The EPP ServiceAccount needs `get`, `list` and `watch` on
`discovery.k8s.io/endpointslices` in `namespace`.

### Examples

```yaml
plugins:
  - type: endpointslice-discovery
    name: slice-discovery
    parameters:
      namespace: default
      selector: kubernetes.io/service-name=model-server
      portName: http
dataLayer:
  discovery:
    endpoints:
      pluginRef: slice-discovery
```

A selector-less headless Service and an EndpointSlice the config above
selects:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-server
spec:
  clusterIP: None
  ports:
    - name: http
      port: 9080
---
apiVersion: discovery.k8s.io/v1
kind: EndpointSlice
metadata:
  name: model-server-d0
  labels:
    kubernetes.io/service-name: model-server
    endpointslice.kubernetes.io/managed-by: example-controller
    llm-d.ai/role: decode
addressType: IPv6
ports:
  - name: http
    port: 9080
endpoints:
  - addresses: ["2001:db8::20"]
    hostname: d0
    conditions:
      ready: true
```

## Limitations

- The plugin reads the selected slices only. It does not resolve a Service's
  Pod selector or read Pods.
- Metrics are scraped from `address:port`; separate metrics endpoints are not
  supported.
- Labels are read per slice, so every endpoint in a slice gets the same
  labels. Publish one slice per endpoint when endpoints need different labels,
  for example different roles.
- Terminating endpoints are not reported even when their `serving` condition
  is true.
- The EPP runs without a Kubernetes controller manager in this mode, so the
  same K8s-only features as in file discovery are inactive (see
  [docs/discovery.md](../../../../../../../docs/discovery.md)).

## Related Documentation

- [Endpoint Discovery](../../../../../../../docs/discovery.md)
- [File Discovery Plugin](../file/README.md)
- [Plugins Index](../../../README.md)
