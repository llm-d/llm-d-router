# MoRI-IO WRITE-mode and Wide-EP Feature (DORMANT)

> **Status: DORMANT - Not Yet Enabled**
>
> This feature is included in the codebase but is **not yet enabled** for production use.
> Attempting to use `--moriio-write-mode` or related flags will result in an error.

## Overview

This code adds support for AMD MoRI-IO WRITE-mode and Wide-EP (Expert Parallelism) 
disaggregation topologies to the llm-d sidecar. The feature enables:

- **MoRI-IO WRITE-mode**: Prefill RDMA-writes KV cache directly to decode pods
- **Wide-EP DP-rank pinning**: Deterministic routing of requests to specific DP ranks
- **Multi-pod fan-out (2P2D)**: Support for DP=EP=16 across 2 prefill + 2 decode pods

## Why is this feature dormant?

The feature code has been merged to:
1. Allow early review and feedback
3. Enable incremental CI test development

However, full production validation and CI integration are still in progress.
The feature will be enabled in a future Release Candidate (RC) after:
- [ ] End-to-end CI tests are added
- [ ] Production deployment validation is complete
- [ ] Documentation and deployment guides are finalized

## How to enable (FUTURE)

Once the feature is ready for release, the constant in `options.go` will be changed:

```go
// Current (dormant):
MoRIIOFeatureEnabled = false

// Future (enabled):
MoRIIOFeatureEnabled = true
```

## Related Flags (currently blocked)

The following flags are reserved for this feature and will error if used:

| Flag | Purpose |
|------|---------|
| `--moriio-write-mode` | Enable MoRI-IO WRITE-mode |
| `--moriio-parallel-dispatch` | Concurrent prefill/decode dispatch |
| `--moriio-dp-size` | Data parallel world size |
| `--moriio-dp-size-local` | Per-pod DP size for multi-pod |
| `--moriio-remote-hosts` | Prefill pod IPs for decode fan-out |
| `--moriio-decode-hosts` | Decode pod IPs for prefill fan-out |

## Contact

For questions about this feature or its timeline, please contact:
- AMD team
- llm-d maintainers

## Related PRs and Issues

- PR #1564: Initial MoRI-IO WRITE-mode + Wide-EP implementation
