# Exos PoC — step 1: EPP discovery from Exos EndpointSlices

Question: can a standalone llm-d EPP, using the custom-discovery extension point
(`dataLayer.discovery.endpoints`, see `docs/discovery.md`), discover Borneo
linked-runner model servers from the EndpointSlices ECP publishes?

**Answer: yes.** `go test ./exos/poc/` passes against a real kube-apiserver.

## Layout

| Path | What |
|---|---|
| `discovery/endpointslice/` | `exos-endpointslice-discovery` plugin (`fwkdl.EndpointDiscovery`): informer on EndpointSlices → `DiscoveryNotifier` Upsert/Delete; label mapping (`exos/role` → `llm-d.ai/role`), static labels, `StateDumper` |
| `cmd/exos-epp/` | Stock EPP runner + out-of-tree registration of the plugin |
| `poc/simecp/` | Reproduces the Exos cluster: control node with `linked-runner.runner-instance/<runner>` annotations, hostNetwork model pods (podIP = control node IP), and ECP's headless Services + EndpointSlices (same fields as `ecp/server_main.go applyEndpointSliceInNamespace`) |
| `poc/poc_test.go` | envtest apiserver + simecp + real `exos-epp` subprocess + Envoy-style ext_proc client |

## Run

```bash
# once: envtest binaries (kube-apiserver + etcd)
go install sigs.k8s.io/controller-runtime/tools/setup-envtest@release-0.25
export KUBEBUILDER_ASSETS=$(setup-envtest use 1.36.2 -p path)

go test -race ./exos/discovery/...     # unit tests (fake clientset)
go test -v ./exos/poc/                 # end-to-end
```

## What the e2e test verifies

1. Pod-based view (EPP default): all three model pods report `podIP=10.128.0.5`
   (control node) → unusable for discovery.
2. With the plugin, the EPP datastore (`/debug/plugins/state` on the metrics
   port) holds the 3 runners with their **Borneo IPv6** addresses, port 9080,
   `llm-d.ai/role` and `llm-d.ai/engine-type=sglang` labels, and never the
   control node IP.
3. ext_proc (`/v1/chat/completions`, profile = `decode-filter` + `random-picker`)
   returns `x-gateway-destination-endpoint: [2606:ec80:10:263d::20]:9080` or
   `...::21]:9080` — only decode runners, IPv6 correctly bracketed.
4. Setting a runner's slice `ready=false` removes it from the datastore and from
   routing.

## Findings to carry into ECP / later steps

- **Readiness**: today's ECP hard-codes `ready/serving=true` on the slice
  (runner RUNNING ≠ model ready). The plugin honors the condition; ECP must set
  it from `exos.google.com/model-ready` (or use proxy-pod readiness instead).
- **Role**: today's ECP slices have no `exos/role`. `decode-filter` *accepts
  endpoints without a role*, so with today's slices every runner (including
  prefill) would be a decode candidate. The PoC adds `exos/role` to the slices
  (`simecp.Reconcile(..., roleLabels=true)`); ECP needs the same change.
- Custom-discovery mode starts no controller manager, so the plugin builds its
  own client (in-cluster config, `parameters.kubeconfig`, or `$KUBECONFIG`).
  The EPP ServiceAccount needs `get/list/watch` on `discovery.k8s.io/endpointslices`.
- The plugin type is registered as Alpha → run EPP with `--allow-experimental-plugins`.
- Not covered yet: metrics scraping (IPv6 targets are unreachable in the test;
  in Exos it needs mTLS or `MetricsHost` → proxy pod), Envoy, routing-proxy/PD.
