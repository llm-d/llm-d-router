# Peer-to-Peer Syncer

**Type:** `p2p-syncer`
**Interface:** `CrossReplicaSyncer`, `CrossReplicaBatchSyncer`, `PeerConsumer`

Shares endpoint state between EPP replicas by sending it directly to each
peer over gRPC, with no external store. Each EPP publishes its local
endpoint value to every discovered peer. Received values update the peer
aggregate in the EPP's in-process cache as they arrive. `Get` reads the
contributor's live local value and combines it with the cached peer
aggregate.

The plugin does not support request-level coordination: `GetOrSet` returns
`ErrCoordinationUnsupported`.

## Configuration

Enable this Alpha plugin with `--allow-experimental-plugins=true`. The plugin
needs peer discovery to find its peers.

```yaml
plugins:
  - type: k8s-peer-discovery
    name: peers
    parameters:
      selector: "app=my-router-epp"
      port: "9005"
      namespace: my-namespace
  - type: p2p-syncer
    name: p2p
    parameters:
      port: "9005"
      stateTTL: 2s
      refreshInterval: 100ms

dataLayer:
  discovery:
    peers:
      pluginRef: peers
  crossReplica:
    syncerPluginRef: p2p
    syncInterval: 20ms
```

Parameters:

- `port`: TCP port this EPP listens on for peer state. Required. It must match
  the `port` of the peer discovery plugin, which tells each EPP where to send.
- `stateTTL`: How long a peer's value counts after the last message that
  carried it. Defaults to `2s`.
- `refreshInterval`: Longest time between two sends of an unchanged value.
  Defaults to `100ms` and must be shorter than `stateTTL`.

### Publisher timing

Publisher settings are configured under `dataLayer.crossReplica`, as for any
syncer. The plugin supports batching, so the publisher calls it once per
contributor per `syncInterval` tick with the live endpoint set. Each call reads
every endpoint's local value and sends one message per peer with the values
whose encoding changed since they were last sent, plus those last sent
`refreshInterval` or longer ago. A tick with no changes sends nothing, so `syncInterval` can be much
shorter than with a store-backed syncer: a peer sees a change one `syncInterval`
plus one network hop after it happens. Sending never waits on a peer, and
`publishTimeout` is not used.

A replica with nothing to send for `refreshInterval` sends an empty message so
its peers can tell it is reachable.

A value is dropped when no message refreshes it for `stateTTL`, so a `stateTTL`
several times `refreshInterval` lets a value survive a few lost messages.

## State Model

Each EPP keeps, per state key and endpoint, the last value received from each
peer replica and when it arrived. A replica is identified by its hostname and a
process-specific UUID. When a message arrives, the EPP decodes its values,
recomputes the peer aggregate for each endpoint in it, and replaces the cached
aggregate. A value that is not refreshed within `stateTTL` is dropped and the
aggregate recomputed, so a peer that stops sending stops counting `stateTTL`
after its last message. `Delete` sends peers a message to drop this replica's
value for the endpoint; a peer that misses it drops the value after `stateTTL`.
`Get` reads the cache without network access and combines it with the
contributor's live local value. If no peer value is available, `Get` returns the
local value.

Values are encoded with `encoding/gob` and decoded into the type supplied by
the local contributor, as with the Redis syncer. An EPP keeps a peer's value
only for an endpoint it has sent itself under the same state key. A value that
arrives before that is dropped, and the peer sends it again within
`refreshInterval`. An EPP holds values from at most 64 peer replicas at once;
messages from further replicas are dropped until a held replica goes `stateTTL`
without a message.

## Connections

Each EPP serves one gRPC listener on `port` and keeps one client connection
and one stream per discovered peer, on which it sends its messages; the peer
never replies. A peer that stops reading its stream blocks only the link to
that peer. The link queues at most 256 messages and drops new ones beyond that;
each value is resent within `refreshInterval` of the link draining.

A message larger than 16 MiB is not sent, and a received one ends its stream.
A received message that does not decode is skipped, and its stream stays open.
Each incoming connection is set up on its own with a 5 s limit, so a client
that connects and sends nothing holds only its own connection. An EPP accepts
at most 64 connections, with at most 2 streams on each.

The peer set comes from peer discovery and is read every 50 ms:

- A new peer gets a link, and each time a link's stream opens every value is
  resent to all peers on the next publish.
- A peer that leaves the peer set loses its link. Its values are not dropped
  with it: a replica that is draining after SIGTERM leaves the peer set while it
  still serves requests, keeps sending, and stops counting `stateTTL` after its
  last message.
- A lost connection is redialed with backoff of at most 1 s, so a restarted
  peer on the same address gets every value within about a second.
- A connection on which the peer acknowledges nothing for 15 s, 10 s of
  keepalive time plus 5 s of timeout, is closed and redialed. This replaces a
  connection that stopped delivering after a network partition.

Opening, connecting, losing and closing a link, a link not connected for
`stateTTL`, the first message from a peer replica, and a peer replica going
`stateTTL` without a message are logged at the default level. After the first
64 peer replicas, a new one is logged at most once every 30 s.

## Scheduling and Admission

Periodic endpoint synchronization provides peer-aware scheduling state. It is
eventually consistent: two EPP replicas can make scheduling or admission
decisions before either observes the other's latest work. The window is one
`syncInterval` plus one network hop.

This synchronization does not implement an atomic cluster-wide admission quota.

`GetOrSet` returns `ErrCoordinationUnsupported`, because there is no store that
every replica writes through and so no linearizable request-level value. A
plugin that calls `GetOrSet` must not be configured to coordinate across
replicas with this syncer. For `disaggregatedset-rollout-screener`, set
`disableCoordination: true`; with coordination on, the screener returns no
endpoints when `GetOrSet` fails.

## Deployment

The plugin needs no server. Every EPP replica must be able to reach every other
replica's `port` over TCP, and the replicas must be in the peer set of the
discovery plugin. Connections use unencrypted TCP with no authentication, so
`port` should be reachable only from EPP pods, for example through a
NetworkPolicy.

The plugin binds `port` when it is created, so a port in use fails EPP startup.
Peers that are not reachable yet do not prevent startup: their links connect
when the peer listens. Before the first message from a peer an EPP counts local
state only for it.

During a network partition each side drops the other side's values `stateTTL`
after the last message, and then counts local state only for the replicas it
cannot reach. After a partition longer than 15 s, values return within about a
second of the network healing. After a shorter one they return when TCP
retransmits on the open connection, which can take several seconds.

## Related Documentation

- [Plugins Index](../../../README.md)
- [Peer Discovery](../../../../../../../docs/peer-discovery.md)
