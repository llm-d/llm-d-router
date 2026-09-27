# Peer Discovery

## Table of Contents

- [Overview](#overview)
- [The PeerDiscovery Interface](#the-peerdiscovery-interface)
  - [PeerDiscovery](#peerdiscovery)
  - [PeerNotifier](#peernotifier)
  - [PeerMetadata](#peermetadata)
  - [Ordering contract](#ordering-contract)
  - [PeerStore](#peerstore)
- [In-tree plugin: k8s-peer-discovery](#in-tree-plugin-k8s-peer-discovery)
- [Selecting a peer discovery plugin in the EPP config](#selecting-a-peer-discovery-plugin-in-the-epp-config)
- [Writing a peer discovery plugin](#writing-a-peer-discovery-plugin)

---

## Overview

EPP replicas operating in an active-active topology need to know about each
other so a future cross-replica syncer can exchange state (e.g. prefix-cache
metadata) between them. **Peer discovery** is the mechanism that tracks the
set of live EPP replicas and feeds it to consumers through a `PeerNotifier`.

Peer discovery mirrors the [endpoint discovery](discovery.md) plugin model:

- It is configured via `dataLayer.discovery.peers.pluginRef` in the
  `EndpointPickerConfig`.
- Implementations are registered in the plugin registry and selected by name.
- When omitted, peer discovery is disabled and no peer set is maintained.

### Why PeerStore is separate from Datastore

`PeerStore` and `Datastore` share the same Upsert/Delete/notifier shape, but
they serve different consumers and have different lifecycles:

| | Endpoints (Datastore) | Peers (PeerStore) |
|---|---|---|
| **Consumer** | Scheduler, request routing | Cross-replica syncer |
| **Lifecycle** | Bound to the InferencePool | Bound to the EPP Deployment |
| **Identity** | Model-serving pod | EPP replica |

Keeping them separate avoids coupling unrelated subsystems.

---

## The PeerDiscovery Interface

### PeerDiscovery

```go
// pkg/epp/framework/interface/datalayer/peer.go

type PeerDiscovery interface {
    plugin.Plugin   // provides TypedName() TypedName

    // Start begins discovery and blocks until ctx is cancelled or a fatal
    // error occurs. The caller invokes Start in a dedicated goroutine.
    Start(ctx context.Context, notifier PeerNotifier) error

    // Ready returns a channel that is closed once discovery is delivering
    // peers. The peer set may be incomplete when Ready closes and changes
    // over time.
    Ready() <-chan struct{}
}
```

### PeerNotifier

```go
type PeerNotifier interface {
    // Upsert adds or updates a peer.
    Upsert(peer *PeerMetadata)
    // Delete removes a peer by its identity.
    Delete(id types.NamespacedName)
}
```

### PeerMetadata

| Field | Type | Description |
|---|---|---|
| `ID` | `types.NamespacedName` | Stable identity of the peer, unique across the peer set. |
| `Address` | `string` | IP address of the peer replica. |
| `Port` | `string` | Port the peer listens on for state sync. Empty when the discovery source does not expose a port. |

### Ordering contract

`PeerNotifier` is **not goroutine-safe**. All `Upsert` and `Delete` calls
must be made sequentially from a single goroutine so that an `Upsert`
followed by a `Delete` for the same peer is applied in that order.

### PeerStore

`PeerStore` is the narrow interface consumed by `NewPeerNotifier`:

```go
type PeerStore interface {
    PeerUpsert(ctx context.Context, meta *PeerMetadata)
    PeerDelete(id types.NamespacedName)
}
```

The in-tree `MemoryPeerStore` (`pkg/epp/statesync/peerstore.go`) is a
goroutine-safe, in-memory implementation that also exposes a `Peers()` method
returning a deterministically ordered snapshot of the current peer set.

---

## In-tree plugin: k8s-peer-discovery

The `k8s-peer-discovery` plugin
(`pkg/epp/framework/plugins/datalayer/discovery/k8speer`) registers a Pod
notification extractor with the datalayer runtime. The runtime owns the
controller-runtime watch and runs it on every EPP replica. The plugin filters
Pods by namespace, label selector, readiness, and self IP. Each ready,
non-self Pod is upserted through the `PeerNotifier` passed to `Start`; Pods
that become unready or are deleted are removed. `Start` applies all events
from its own goroutine, so the notifier's single-goroutine contract holds.

The plugin closes `Ready()` when `Start` runs. Peers from the initial Pod
list may arrive after that.

### Parameters

| Parameter | Type | Required | Description |
|---|---|---|---|
| `selector` | `string` | yes | Kubernetes label selector matching this EPP Deployment's pods (e.g. `app=my-epp`). |
| `port` | `string` | yes | Port peer replicas listen on for state sync. Pods do not self-report a service port, so it comes from config. Values must be in the range 1 through 65535. |
| `namespace` | `string` | yes | Namespace of this EPP Deployment's pods. It must match the EPP deployment namespace because the Pod cache is scoped to that namespace. When the `NAMESPACE` environment variable is set, the plugin validates the value at startup. |

The plugin uses `POD_IP` to exclude this replica from the peer set. When it is
unset, the plugin uses the Pod hostname instead. If `NAMESPACE` is unset, the
plugin logs that it cannot validate the configured namespace.

### Example config

```yaml
plugins:
  - name: peer-disc
    type: k8s-peer-discovery
    parameters:
      selector: "app=my-epp"
      port: "9002"
      namespace: default

dataLayer:
  discovery:
    peers:
      pluginRef: peer-disc
```

---

## Selecting a peer discovery plugin in the EPP config

Add a `peers` entry inside `dataLayer.discovery` in the
`EndpointPickerConfig`. The `pluginRef` field names a plugin instance defined
in the top-level `plugins` list.

```yaml
plugins:
  - name: my-peer-disc
    type: k8s-peer-discovery
    parameters:
      selector: "app=my-epp"
      port: "9002"
      namespace: default

dataLayer:
  discovery:
    peers:
      pluginRef: my-peer-disc  # must match a name in plugins above
```

When `dataLayer.discovery.peers` is absent, peer discovery is disabled and no
peer set is maintained (backwards compatible).

---

## Writing a peer discovery plugin

Implement `fwkdl.PeerDiscovery` and register the factory with the plugin
registry. A runnable example of this pattern is in the test suite:

- [`pkg/epp/framework/interface/datalayer/peer_test.go`](../pkg/epp/framework/interface/datalayer/peer_test.go)
  tests the interface contract with a fake store.
- [`pkg/epp/statesync/peerstore_test.go`](../pkg/epp/statesync/peerstore_test.go)
  (`TestPeerDiscoveryFullWiring`) exercises the full plugin-to-store pipeline
  with `MemoryPeerStore`.

```go
package myplugin

import (
    "context"
    "encoding/json"
    "sync"

    "k8s.io/apimachinery/pkg/types"

    fwkdl    "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
    fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const PluginType = "my-peer-discovery"

type MyPeerDiscovery struct {
    typedName fwkplugin.TypedName
    ready     chan struct{}
    readyOnce sync.Once
}

var _ fwkdl.PeerDiscovery = (*MyPeerDiscovery)(nil)

func Factory(name string, params *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
    if name == "" {
        name = PluginType
    }
    return &MyPeerDiscovery{
        typedName: fwkplugin.TypedName{Type: PluginType, Name: name},
        ready:     make(chan struct{}),
    }, nil
}

func (d *MyPeerDiscovery) TypedName() fwkplugin.TypedName { return d.typedName }
func (d *MyPeerDiscovery) Ready() <-chan struct{}           { return d.ready }

func (d *MyPeerDiscovery) Start(ctx context.Context, notifier fwkdl.PeerNotifier) error {
    // 1. Enumerate existing peers.
    notifier.Upsert(&fwkdl.PeerMetadata{
        ID:      types.NamespacedName{Name: "epp-1", Namespace: "default"},
        Address: "10.0.0.2",
        Port:    "9002",
    })
    d.readyOnce.Do(func() { close(d.ready) })

    // 2. Watch for changes and call Upsert/Delete as peers come and go.
    //    All calls must be from this goroutine (PeerNotifier is not
    //    goroutine-safe).
    <-ctx.Done()
    return nil
}
```

Register the factory in the runner:

```go
fwkplugin.Register(myplugin.PluginType, fwkplugin.StabilityAlpha, myplugin.Factory)
```

Then reference it in the EPP config:

```yaml
plugins:
  - name: my-peer-disc
    type: my-peer-discovery
    parameters: {}

dataLayer:
  discovery:
    peers:
      pluginRef: my-peer-disc
```
