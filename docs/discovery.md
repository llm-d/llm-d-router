# Endpoint Discovery

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [The EndpointDiscovery Interface](#the-endpointdiscovery-interface)
  - [EndpointDiscovery](#endpointdiscovery)
  - [DiscoveryNotifier](#discoverynotifier)
  - [Ordering contract](#ordering-contract)
  - [Wiring to the datastore](#wiring-to-the-datastore)
- [Selecting a discovery plugin in the EPP config](#selecting-a-discovery-plugin-in-the-epp-config)
- [File Discovery plugin](#file-discovery-plugin)
  - [Parameters](#parameters)
  - [Endpoints file format](#endpoints-file-format)
  - [Live reload](#live-reload)
  - [Validation rules](#validation-rules)
- [Running EPP with file discovery (no Kubernetes)](#running-epp-with-file-discovery-no-kubernetes)
  - [What you need](#what-you-need)
  - [1. Endpoints file](#1-endpoints-file)
  - [2. EPP config](#2-epp-config)
  - [3. Start the EPP](#3-start-the-epp)
  - [4. Envoy config](#4-envoy-config)
  - [5. Start Envoy](#5-start-envoy)
- [Writing a custom discovery plugin](#writing-a-custom-discovery-plugin)

---

## Overview

By default the EPP discovers inference endpoints by watching a Kubernetes
`InferencePool` object and the pods it selects.  The **discovery plugin
interface** replaces that mechanism with a pluggable alternative so the EPP
can run in environments where Kubernetes is not available -- for example bare
metal inference clusters, Slurm jobs, or local development.

When a discovery plugin is configured:

- The EPP **does not** call `ctrl.GetConfig()` or start a controller manager.
- The EPP **does not** require an `InferencePool` CRD or any K8s RBAC.
- The discovery plugin is solely responsible for populating the endpoint
  datastore via `DiscoveryNotifier`.
- All scheduling, request control, and metrics collection behaviour is
  unchanged.

---

## Architecture

```
  Client request
       |
       v
  +--------+          +------------------------------+
  |  Envoy |--ext_proc-->        EPP (port 9002)      |
  +--------+  gRPC    |                              |
       |               |  Scheduler  |  Datastore    |
       |               |             |   endpoints   |
       |               |          ^  |               |
       |               +----------|--+               |
       |                          |
       |               +----------+----------+
       |               |   EndpointDiscovery  |
       |               |   plugin             |
       |               |                      |
       |               |  (file-discovery)    |
       |               |  reads endpoints.yaml|
       |               |  calls Upsert/Delete |
       |               +---------------------+
       |
       | x-gateway-destination-endpoint header set by EPP
       v
  selected inference endpoint (e.g. 10.0.0.1:8000)
```

The EPP sets the `x-gateway-destination-endpoint` header on the response to
`ext_proc`.  Envoy's `ORIGINAL_DST` cluster reads that header and forwards
the request to the chosen endpoint.

---

## The EndpointDiscovery Interface

### EndpointDiscovery

```go
// pkg/epp/framework/interface/datalayer/discovery.go

type EndpointDiscovery interface {
    plugin.Plugin   // provides TypedName() TypedName

    // Start begins discovery and blocks until ctx is cancelled or a fatal
    // error occurs. It must be called in a dedicated goroutine.
    //
    // Implementations SHOULD enumerate all currently known endpoints via
    // notifier.Upsert before entering the watch loop to avoid serving an
    // empty datastore at startup.
    Start(ctx context.Context, notifier DiscoveryNotifier) error
}
```

### DiscoveryNotifier

```go
type DiscoveryNotifier interface {
    // Upsert adds or updates an endpoint in the datastore.
    Upsert(endpoint *EndpointMetadata)
    // Delete removes an endpoint by its namespaced name.
    Delete(id types.NamespacedName)
}
```

`EndpointMetadata` fields relevant to discovery:

| Field | Type | Description |
|---|---|---|
| `NamespacedName` | `types.NamespacedName` | Unique identity of the endpoint. |
| `PodName` | `string` | Logical name (used in metrics). |
| `Address` | `string` | IP address of the inference server. |
| `Port` | `string` | Port as a string (e.g. `"8000"`). |
| `MetricsHost` | `string` | `host:port` for metrics scraping. Defaults to `address:port` if empty. |
| `Labels` | `map[string]string` | Arbitrary labels available to scheduler plugins. |

### Ordering contract

`DiscoveryNotifier` is **not goroutine-safe**.  All `Upsert` and `Delete`
calls must be made sequentially from a single goroutine and in causal order.
An `Upsert` followed by a `Delete` for the same endpoint must arrive in that
order or the endpoint will be incorrectly left in the datastore.

### Wiring to the datastore

The runner creates a `DiscoveryNotifier` backed by the datastore and passes
it to `EndpointDiscovery.Start`:

```go
disc.Start(ctx, fwkdl.NewDiscoveryNotifier(ds))
```

`NewDiscoveryNotifier` accepts any value that implements
`DiscoveryBackendStore` (i.e. has `BackendUpsert` and `BackendDelete`),
which the production `Datastore` satisfies.

---

## Selecting a discovery plugin in the EPP config

Add a `discovery` section inside `dataLayer` in the `EndpointPickerConfig`.
The `pluginRef` field names a plugin instance defined in the top-level
`plugins` list.

```yaml
plugins:
  - name: my-endpoints          # the name referenced by pluginRef below
    type: file-discovery        # plugin type registered in the runner
    parameters:
      path: /etc/epp/endpoints.yaml
      watchFile: true

# ... other plugins (scorers, filters, etc.) ...

dataLayer:
  discovery:
    pluginRef: my-endpoints     # must match a name in plugins above
```

When `dataLayer.discovery` is present the EPP skips the Kubernetes setup
entirely.  When it is absent the EPP uses the default Kubernetes-based
discovery (backwards compatible).

---

## File Discovery plugin

**Plugin type:** `file-discovery`

Reads a static YAML or JSON file that lists inference endpoints, calls
`DiscoveryNotifier.Upsert` for each entry, and optionally watches the file
for changes using `fsnotify`.

### Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `path` | string | yes | -- | Absolute or relative path to the endpoints file. |
| `watchFile` | bool | no | `false` | Re-read the file and reconcile the datastore whenever the file is written or replaced. |

### Endpoints file format

The file may be YAML or JSON.  YAML is recommended for readability.

```yaml
endpoints:
  - name: <string>              # required -- unique within the file
    namespace: <string>         # optional -- defaults to "default"
    address: <IPv4 or IPv6>     # required -- must be a valid IP address
    port: <string>              # required -- integer 1-65535 as a string
    metricsHost: <string>       # optional -- host:port for metrics scraping
                                #            defaults to address:port
    labels:                     # optional -- arbitrary key/value labels
      <key>: <value>
```

Example with two vLLM instances:

```yaml
endpoints:
  - name: vllm-0
    address: "10.0.0.1"
    port: "8000"
    labels:
      model: llama-3-8b

  - name: vllm-1
    address: "10.0.0.2"
    port: "8000"
    metricsHost: "10.0.0.2:9090"   # scrape metrics from a different port
    labels:
      model: llama-3-8b
```

Example with multi-rank (tensor-parallel) setup where each rank is a
separate endpoint:

```yaml
endpoints:
  - name: tp-rank-0
    namespace: inference
    address: "192.168.1.10"
    port: "8000"
    labels:
      rank: "0"
      job: tp-job-42

  - name: tp-rank-1
    namespace: inference
    address: "192.168.1.11"
    port: "8000"
    labels:
      rank: "1"
      job: tp-job-42
```

**Constraints:**

- `address` must parse as a valid IP address (`net.ParseIP`).
- `port` must be a decimal integer in the range 1-65535.
- The file must not exceed 1 MiB.
- Duplicate names within the same namespace result in the last `Upsert`
  winning (the file is processed top-to-bottom).

### Live reload

When `watchFile: true` the plugin watches the file via `fsnotify`.  On a
`Write` or `Create` event it re-reads the file, upserts all current
endpoints, and deletes any endpoint that was present in the previous load
but is absent from the new file.  Reload errors are logged but do not
terminate the plugin -- the datastore retains its previous state.

Atomic file replacement (write to a temp file then rename) triggers a
`Create` event and is handled correctly.

### Validation rules

The plugin fails at startup (before `Start` returns an error) if:

- `path` parameter is missing or empty.
- The file cannot be opened.
- The file exceeds 1 MiB.
- The YAML/JSON cannot be parsed.
- Any endpoint has an invalid `address` or `port`.

---

## Running EPP with file discovery (no Kubernetes)

This example runs the EPP alongside Envoy on a single machine with two
vLLM processes.  No Kubernetes, no `InferencePool` CRD.

```
localhost:8080  <-- Envoy (user-facing)
localhost:9002  <-- EPP gRPC (ext_proc)
localhost:9003  <-- EPP health (gRPC)
localhost:9090  <-- EPP Prometheus metrics

10.0.0.1:8000   <-- vLLM instance 0
10.0.0.2:8000   <-- vLLM instance 1
```

### What you need

- `llm-d-inference-scheduler` binary built from this repo.
- Envoy v1.31+ binary or Docker image.
- vLLM (or a compatible inference server) running on the target addresses.

### 1. Endpoints file

Save as `/etc/epp/endpoints.yaml`:

```yaml
endpoints:
  - name: vllm-0
    address: "10.0.0.1"
    port: "8000"

  - name: vllm-1
    address: "10.0.0.2"
    port: "8000"
```

### 2. EPP config

Save as `/etc/epp/config.yaml`.  This example uses the default KV-cache
utilization scorer and random picker -- adapt scheduling plugins as needed.

```yaml
plugins:
  - name: file-disc
    type: file-discovery
    parameters:
      path: /etc/epp/endpoints.yaml
      watchFile: true

  - name: kv-cache-scorer
    type: kv-cache-utilization-scorer

  - name: random-picker
    type: random-picker

  - name: single-profile
    type: single-profile-handler
    parameters:
      scorer: kv-cache-scorer
      picker: random-picker

schedulingProfiles:
  - name: default
    plugins:
      - pluginRef: single-profile

dataLayer:
  discovery:
    pluginRef: file-disc
```

### 3. Start the EPP

```bash
llm-d-inference-scheduler \
  --pool-name epp \
  --config-file /etc/epp/config.yaml \
  --grpc-port 9002 \
  --grpc-health-port 9003 \
  --metrics-port 9090
```

`--pool-name` is required even in file discovery mode; it is used as the
internal pool identifier in the scheduler and metrics.  The value is
arbitrary -- it does not need to match any Kubernetes object.

The EPP logs `EPP starting (file discovery mode)` on startup and emits a
log line for each endpoint loaded from the file.

### 4. Envoy config

Save as `/etc/envoy/envoy.yaml`.  The key pieces are:

- A listener on `0.0.0.0:8080` that applies the `ext_proc` filter.
- The `ext_proc` filter pointing at the EPP on `localhost:9002`.
- An `ORIGINAL_DST` cluster that reads the `x-gateway-destination-endpoint`
  header written by the EPP and forwards to the chosen endpoint.

```yaml
admin:
  address:
    socket_address:
      address: 127.0.0.1
      port_value: 19000

static_resources:
  listeners:
    - name: inference
      address:
        socket_address:
          address: 0.0.0.0
          port_value: 8080
      filter_chains:
        - filters:
            - name: envoy.filters.network.http_connection_manager
              typed_config:
                "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                stat_prefix: inference
                route_config:
                  name: inference
                  virtual_hosts:
                    - name: inference
                      domains: ["*"]
                      routes:
                        - match:
                            prefix: "/"
                          route:
                            cluster: original_destination_cluster
                            timeout: 600s
                            idle_timeout: 600s
                http_filters:
                  - name: envoy.filters.http.ext_proc
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.ext_proc.v3.ExternalProcessor
                      grpc_service:
                        envoy_grpc:
                          cluster_name: epp
                          authority: localhost:9002
                        timeout: 10s
                      processing_mode:
                        request_header_mode: SEND
                        response_header_mode: SEND
                        request_body_mode: FULL_DUPLEX_STREAMED
                        response_body_mode: FULL_DUPLEX_STREAMED
                        request_trailer_mode: SEND
                        response_trailer_mode: SEND
                      message_timeout: 600s
                  - name: envoy.filters.http.router
                    typed_config:
                      "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router

  clusters:
    - name: epp
      type: STATIC
      connect_timeout: 86400s
      lb_policy: LEAST_REQUEST
      typed_extension_protocol_options:
        envoy.extensions.upstreams.http.v3.HttpProtocolOptions:
          "@type": type.googleapis.com/envoy.extensions.upstreams.http.v3.HttpProtocolOptions
          explicit_http_config:
            http2_protocol_options: {}
      load_assignment:
        cluster_name: epp
        endpoints:
          - lb_endpoints:
              - endpoint:
                  address:
                    socket_address:
                      address: 127.0.0.1
                      port_value: 9002

    - name: original_destination_cluster
      type: ORIGINAL_DST
      connect_timeout: 600s
      lb_policy: CLUSTER_PROVIDED
      circuit_breakers:
        thresholds:
          - max_connections: 10000
            max_pending_requests: 10000
            max_requests: 10000
      original_dst_lb_config:
        use_http_header: true
        http_header_name: x-gateway-destination-endpoint
```

### 5. Start Envoy

```bash
envoy -c /etc/envoy/envoy.yaml
```

Requests to `http://localhost:8080/v1/completions` are now routed through
the EPP to one of the two vLLM instances.

---

## Writing a custom discovery plugin

Implement `fwkdl.EndpointDiscovery` and register the factory with the
runner's plugin registry.

```go
package myplugin

import (
    "context"
    "encoding/json"
    "fmt"

    "k8s.io/apimachinery/pkg/types"

    fwkdl    "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
    fwkplugin "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
)

const PluginType = "my-discovery"

type MyDiscovery struct {
    typedName fwkplugin.TypedName
    // ... plugin state ...
}

var _ fwkdl.EndpointDiscovery = (*MyDiscovery)(nil)

func Factory(name string, params json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
    if name == "" {
        name = PluginType
    }
    return &MyDiscovery{
        typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
    }, nil
}

func (d *MyDiscovery) TypedName() fwkplugin.TypedName { return d.typedName }

func (d *MyDiscovery) Start(ctx context.Context, notifier fwkdl.DiscoveryNotifier) error {
    // 1. Enumerate existing endpoints.
    notifier.Upsert(&fwkdl.EndpointMetadata{
        NamespacedName: types.NamespacedName{Name: "ep0", Namespace: "default"},
        Address:        "10.0.0.1",
        Port:           "8000",
        MetricsHost:    "10.0.0.1:8000",
    })

    // 2. Watch for changes and call Upsert/Delete as endpoints come and go.
    //    All calls must be from this goroutine (DiscoveryNotifier is not goroutine-safe).
    <-ctx.Done()
    return nil
}
```

Register the factory in the runner before calling `Run`:

```go
fwkplugin.Register(myplugin.PluginType, myplugin.Factory)
```

Then reference it in the EPP config:

```yaml
plugins:
  - name: my-disc
    type: my-discovery
    parameters: {}

dataLayer:
  discovery:
    pluginRef: my-disc
```
