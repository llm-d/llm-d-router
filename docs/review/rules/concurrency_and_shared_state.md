# Concurrency and shared state

## Keep mutex critical sections tight and release locks before network or blocking calls
Encapsulate synchronization on the state-owning type, finish struct initialization before registering callbacks, and never hold a mutex across network RPCs, channel sends, or socket teardown. Combine read-modify-write steps and coupled counter reads under one lock acquisition, and copy caller-owned maps before asynchronous dispatch.
- Accept: Defensive slice or map copies at goroutine boundaries, snapshot coalescing in capacity-one channels, uncontended startup or shutdown locks, and locked handle-invalidation loops when unlocking early races with removal.

## Bound in-memory state with capacity limits and endpoint deletion cleanup
Bound in-memory maps and queues keyed by requests, sessions, prompts, or endpoints with hard capacity caps and periodic TTL sweeps. Subscribe stateful plugins to endpoint lifecycle events so pod removals and empty controller reconciles purge per-endpoint state.
- Accept: A dedicated bounded registry when state requires no-eviction pinning of in-flight entries, or a periodic cleanup loop when missed lifecycle events could strand entries.

## Reconcile dynamic priority bands on controller edges and drain before removal
Add and remove dynamic `InferenceObjective` priority bands on controller reconciliation edges rather than data-plane idleness sweeps, preserving static default bands and draining in-flight queues before band deletion.

## Propagate caller contexts and bound goroutine lifetimes to server drain
Root background loops on a drain-scoped context, pass caller request contexts to all outbound RPCs, and run graceful readiness drain on leader lease loss. Check `ctx.Err()` across loop iterations and multi-channel `select` wakeups so client disconnects halt downstream work and release in-flight reservations.

## Scope cross-replica state to per-replica keys with short remote-store TTLs
Publish cross-replica load and cache state under collision-free replica IDs, combining live local counters with cached peer levels at read time so replica restarts never overwrite peer state. Expire peer entries via short remote-store TTLs scaled to the publish interval instead of host wall-clock comparisons, and keep startup non-blocking when optional peer stores are unreachable.

## Recover KV-event sequence gaps and epoch resets from base snapshots
Track per-endpoint KV-event sequence numbers to detect gaps or worker restarts (`seq < lastSeq`), resetting stale index entries and loading a base snapshot before applying subsequent events.
