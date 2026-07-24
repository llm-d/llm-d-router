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

**Which to use?** Serial is the safe default and gives the tightest correctness guarantee
(decode is pinned to the rank prefill truly used). Parallel reduces end-to-end latency by
overlapping the two legs, at the cost of pinning the rank before prefill confirms it.

## Supported Topologies

| Topology | DP | EP | TP | Pods | Description |
|----------|----|----|----|----- |-------------|
| **1P1D** | 8  | 8  | 1  | 1 prefill + 1 decode | Intra-node Wide-EP |
| **2P2D** | 16 | 16 | 1  | 2 prefill + 2 decode | Inter-node Wide-EP with LeaderWorkerSet |

## CLI Flags

| Flag | Purpose |
|------|---------|
| `--moriio-write-mode` | Enable MoRI-IO WRITE-mode (serial dispatch by default) |
| `--moriio-parallel-dispatch` | Concurrent prefill/decode dispatch (requires `--moriio-write-mode`) |
| `--moriio-dp-size` | Data parallel world size |
| `--moriio-dp-size-local` | Per-pod DP size for multi-pod (`pod_idx = dp_rank / dp_size_local`) |
| `--moriio-remote-hosts` | Prefill-side pod hosts for fan-out (DNS names preferred) |
| `--moriio-decode-hosts` | Decode-side pod hosts, emitted as the prefill leg's `remote_hosts` (DNS names preferred) |
| `--moriio-tp-size` | Tensor parallel size |
| `--moriio-local-pod-ip` | Local pod IP (defaults to `POD_IP` env) |
| `--moriio-decode-handshake-port` | Decode handshake port |
| `--moriio-decode-notify-port` | Decode notify port |
| `--moriio-prefill-handshake-port` | Prefill handshake port |
| `--moriio-prefill-notify-port` | Prefill notify port |

## Host Address Configuration

**DNS hostnames are the recommended and forward-looking way to address pods.** Raw IPs are
still accepted for backward compatibility but are being phased out.

### Recommended: DNS names (LeaderWorkerSet / LWS)

```yaml
# LWS pod names — resolved to IPs at startup
- --moriio-remote-hosts=prefill-master.ns.svc,prefill-worker.ns.svc
- --moriio-decode-hosts=decode-master.ns.svc,decode-worker.ns.svc
```

### Legacy: raw IPs (backward compatibility)

```yaml
# Static IPs (e.g., hostNetwork: true where pod IP = node IP)
- --moriio-remote-hosts=10.0.0.1,10.0.0.2
- --moriio-decode-hosts=10.0.1.1,10.0.1.2
```

**How host resolution works:**
1. At startup, DNS names are resolved to IPs via Kubernetes DNS (IPv4 preferred).
2. Raw IP addresses are passed through unchanged (backward compatibility).
3. Resolution happens once during `Complete()`, before the proxy starts.

**Why DNS:**
- Works with LeaderWorkerSet's predictable naming (`<lws-name>-<group>-<worker>`)
- Enables scalable topologies without hardcoded IP coordination
- Consistent with Kubernetes-native service discovery

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
