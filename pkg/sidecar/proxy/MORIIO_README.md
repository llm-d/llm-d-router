# MoRI-IO WRITE-mode and Wide-EP Feature

> **Status: ENABLED**
>
> This feature is enabled for AMD MoRI-IO WRITE-mode and Wide-EP deployments.

## Overview

This code adds support for AMD MoRI-IO WRITE-mode and Wide-EP (Expert Parallelism)
disaggregation topologies to the llm-d sidecar. The feature enables:

- **MoRI-IO WRITE-mode**: Prefill RDMA-writes KV cache directly to decode pods
- **Serial and parallel dispatch**: two WRITE-mode dispatch strategies (see below)
- **Wide-EP DP-rank pinning**: Deterministic routing of P/D pairs to the same DP rank
- **Multi-pod fan-out (2P2D)**: Support for DP=EP=16 across 2 prefill + 2 decode pods
- **DNS hostname resolution**: Use hostnames (e.g., LWS pod names) instead of hardcoded IPs

## Dispatch modes

MoRI-IO is **off by default** (the sidecar keeps its standard NIXLv2 behavior). Once
`--moriio-write-mode` is set, WRITE mode runs in **serial** dispatch unless you also pass
`--moriio-parallel-dispatch`.

| Mode | Flags | Dispatch | How the decode DP rank is chosen |
|------|-------|----------|----------------------------------|
| Serial WRITE (default) | `--moriio-write-mode` | Prefill first, await its response, then decode | **Propagated** from the prefill leg's returned `remote_dp_rank` (the rank prefill actually ran on); falls back to a stable hash only if the response omits it |
| Parallel WRITE | `--moriio-write-mode --moriio-parallel-dispatch` | Prefill and decode dispatched **concurrently** | **Pinned up front** from config/hash with `remote_dp_rank_override=true` (prefill has not returned yet), so both legs agree without waiting |

**Router-authoritative routing.** In both modes the sidecar and the vLLM MoRI-IO
connector agree on a single DP rank without each side independently hashing:

- **Serial**: the vLLM prefill connector returns the rank it ran on
  (`remote_dp_rank`, `remote_dp_rank_override=true`); the sidecar copies it onto the
  decode leg's `x-data-parallel-rank` header.
- **Parallel**: the sidecar pins the rank itself and sets `remote_dp_rank_override=true`;
  the connector honors that pin on both legs.

**Which to use?** Serial is the **shipped default and the SLA-safe path**: parallel
dispatch is **OFF unless you explicitly pass `--moriio-parallel-dispatch`**, so with
defaults the sidecar always takes the serial `handleNIXLV2` path. Serial gives the
tightest correctness guarantee (decode is pinned to the rank prefill truly used).

Parallel dispatch is an **opt-in latency optimization** that overlaps the two legs.
Because prefill and decode are issued concurrently, it now includes explicit
**prefill-failure handling** so it stays correct and never hangs:

- **Shared cancelable context.** Both legs derive from one cancelable context. If
  prefill returns any non-2xx status (or a transport error, which surfaces as `502`),
  the sidecar cancels that context so decode's in-flight request / KV wait aborts
  immediately instead of waiting for KV that will never arrive.
- **Commit point (buffered decode response).** Decode is dispatched concurrently but
  its HTTP status/headers/body are held until prefill's outcome is known. If prefill
  **fails**, decode's output is discarded and the **prefill error/status** is returned
  to the client (never a bogus `200`). If prefill **succeeds**, decode's response is
  committed and streamed through with its own status, headers, and body/SSE preserved.
- **Bounded KV-wait backstop.** As a last resort, `--moriio-parallel-decode-wait-timeout`
  (default `30s`) bounds how long decode may wait on the prefill outcome; on expiry both
  legs are cancelled and the request fails with `504` rather than hanging. Prefill in
  parallel dispatch is capped to one token, so it normally resolves well within this
  window.

Trade-off: because the client commit is deferred to the commit point, decode's response
is buffered until prefill is confirmed and then flushed/streamed. In practice prefill
(one token) resolves quickly, so streaming resumes almost immediately after the commit
point; the serial path retains fully incremental streaming from the first byte.

## Supported Topologies

| Topology | DP | EP | TP | Pods | Description |
|----------|----|----|----|----- |-------------|
| **1P1D** | 8  | 8  | 1  | 1 prefill + 1 decode | Intra-node Wide-EP |
| **2P2D** | 16 | 16 | 1  | 2 prefill + 2 decode | Inter-node Wide-EP with LeaderWorkerSet |

## CLI Flags

| Flag | Purpose |
|------|---------|
| `--moriio-write-mode` | Enable MoRI-IO WRITE-mode (serial dispatch by default) |
| `--moriio-parallel-dispatch` | Concurrent prefill/decode dispatch; **OFF by default (serial is the SLA-safe path); opt-in only**. Requires `--moriio-write-mode` |
| `--moriio-parallel-decode-wait-timeout` | Backstop for parallel dispatch: how long decode may wait on the prefill outcome/KV before being cancelled so the request fails (`504`) instead of hanging (default `30s`; only used with `--moriio-parallel-dispatch`) |
| `--moriio-dp-size` | Data parallel world size |
| `--moriio-dp-size-local` | Per-pod DP size for multi-pod (`pod_idx = dp_rank / dp_size_local`) |
| `--moriio-remote-hosts` | Prefill-side pod hosts for fan-out (DNS names preferred) |
| `--moriio-decode-hosts` | Decode-side pod hosts, emitted as the prefill leg's `remote_hosts` (DNS names preferred) |
| `--moriio-tp-size` | Tensor parallel size |
| `--moriio-local-pod-ip` | Local pod address, DNS name or IP (defaults to `POD_IP` env); DNS names resolved to IP at startup |
| `--moriio-decode-handshake-port` | Decode handshake port |
| `--moriio-decode-notify-port` | Decode notify port |
| `--moriio-prefill-handshake-port` | Prefill handshake port |
| `--moriio-prefill-notify-port` | Prefill notify port |

## Host Address Configuration

Peer hosts are Kubernetes DNS names (LeaderWorkerSet / LWS):

```yaml
# LWS pod names — resolved to IPs at startup
- --moriio-remote-hosts=prefill-master.ns.svc,prefill-worker.ns.svc
- --moriio-decode-hosts=decode-master.ns.svc,decode-worker.ns.svc
```

A literal IP may be given in place of any DNS name; it is used as-is (no lookup).

**How host resolution works:**
1. At startup (`Complete()`), DNS names are resolved to IPs via Kubernetes DNS
   (IPv4 preferred) to seed the initial peer IPs, before the proxy starts.
2. On the request path, peer DNS names are **re-resolved under a short TTL**
   (default 30s) when building `kv_transfer_params`, so a peer pod that restarts
   with a new IP is picked up without a router restart. Re-resolution serves the
   last-known-good IP on a transient lookup failure, de-duplicates concurrent
   lookups for the same host via singleflight, and serves the cached IP
   immediately once past the TTL while refreshing asynchronously (only the very
   first, cold-start lookup blocks). A literal IP is used as-is (no lookup).

Both the boot-time and request-path lookups share the same IPv4-preference and
the same per-lookup timeout bound (see [Tuning](#tuning-environment-variables)).

**Why DNS:**
- Works with LeaderWorkerSet's predictable naming (`<lws-name>-<group>-<worker>`)
- Enables scalable topologies without hardcoded IP coordination
- Consistent with Kubernetes-native service discovery

## Tuning (environment variables)

The MoRI-IO DNS re-resolution behavior is tunable at runtime via environment
variables. Each is parsed as a **positive integer**; any non-positive or
non-numeric value falls back to the default.

| Env var | Default | Semantics |
|---------|---------|-----------|
| `MORIIO_DNS_RESOLVE_TTL_SECONDS` | `30` | Maximum age of a cached peer IP before the next request-path lookup re-resolves it. Lower = faster pickup of peer IP changes; higher = fewer DNS lookups under steady state. |
| `MORIIO_DNS_RESOLVE_TIMEOUT_SECONDS` | `5` | Per-lookup context deadline for a single DNS resolution (applied to both the boot-time and request-path lookups). Bounds how long a slow/hanging resolver can stall a cold-start lookup; the effective deadline is the smaller of this value and the remaining request-context deadline. |
| `MORIIO_METRICS_ADDR` | _(unset — disabled)_ | Backward-compatible fallback for the metrics listen address (e.g. `:9090`) when `--metrics-port` is not set. The `--metrics-port` flag takes precedence. Kept on a separate address from the data-plane proxy port. |

### Metrics

The preferred way to expose metrics is the first-class **`--metrics-port`** flag
(matching the EPP `--metrics-port` convention); `0` (the default) disables the
endpoint, and any positive port serves metrics at `/metrics` on that port. The
`MORIIO_METRICS_ADDR` env var remains as a backward-compatible fallback and is
only consulted when `--metrics-port` is unset.

When enabled, the following counters are exposed at `GET /metrics` (unlabeled):

| Metric | Meaning |
|--------|---------|
| `moriio_dns_reresolve_total` | Successful request-path DNS re-resolutions (per actual lookup; singleflight-coalesced lookups count once). |
| `moriio_dns_ip_changed_total` | Re-resolutions where a peer resolved to a different IP than the cached value (peer likely restarted). |
| `moriio_dns_lookup_failures_total` | Failed request-path DNS lookups (resolver then serves the last-known-good IP, or the raw spec on cold start). |

## Example Configurations

### 1P1D (Single-pod, DP=EP=8) — serial WRITE (default)

```bash
--moriio-write-mode=true \
--moriio-dp-size=8 \
--moriio-tp-size=1
```

### 1P1D (Single-pod, DP=EP=8) — parallel WRITE

```bash
--moriio-write-mode=true \
--moriio-parallel-dispatch=true \
--moriio-dp-size=8 \
--moriio-tp-size=1
```

### 2P2D (Multi-pod, DP=EP=16, LeaderWorkerSet) — serial WRITE (default)

```bash
# On decode sidecar:
--moriio-write-mode=true \
--moriio-dp-size=16 \
--moriio-dp-size-local=8 \
--moriio-tp-size=1 \
--moriio-remote-hosts=prefill-master.ns.svc,prefill-worker.ns.svc \
--moriio-decode-hosts=decode-master.ns.svc,decode-worker.ns.svc
```

### 2P2D (Multi-pod, DP=EP=16, LeaderWorkerSet) — parallel WRITE

```bash
# On decode sidecar:
--moriio-write-mode=true \
--moriio-parallel-dispatch=true \
--moriio-dp-size=16 \
--moriio-dp-size-local=8 \
--moriio-tp-size=1 \
--moriio-remote-hosts=prefill-master.ns.svc,prefill-worker.ns.svc \
--moriio-decode-hosts=decode-master.ns.svc,decode-worker.ns.svc
```

## Contact

For questions about this feature, please contact:
- AMD team
- llm-d maintainers

## Related PRs and Issues

- PR #1564: Initial MoRI-IO WRITE-mode + Wide-EP implementation
- vLLM PR #45043: companion vLLM MoRI-IO connector (Wide-EP 2P2D, router-authoritative DP-rank routing)
