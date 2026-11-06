# Exos PoC: EPP discovery from Exos EndpointSlices

Question: can a standalone EPP, using the `endpointslice-discovery` plugin
(`pkg/epp/framework/plugins/datalayer/discovery/endpointslice`), discover
Borneo linked-runner model servers from the EndpointSlices ECP publishes?

`go test ./exos/poc/` answers this against a real kube-apiserver.

## Layout

| Path | What |
|---|---|
| `poc/simecp/` | Reproduces the Exos cluster: control node with `linked-runner.runner-instance/<runner>` annotations, hostNetwork model pods (podIP = control node IP), and ECP's headless Services and EndpointSlices (fields follow `ecp/server_main.go applyEndpointSliceInNamespace`) |
| `poc/poc_test.go` | envtest apiserver + simecp + `cmd/epp` subprocess + ext_proc client acting as Envoy |

## Run

```bash
go install sigs.k8s.io/controller-runtime/tools/setup-envtest@release-0.25
export KUBEBUILDER_ASSETS=$(setup-envtest use 1.36.2 -p path)
go test -v ./exos/poc/
```

## What the test checks

1. All model pods report the control node IP as `podIP`, so Pod-based
   discovery cannot tell the runners apart.
2. The EPP datastore (`/debug/plugins/state`) holds one endpoint per runner.
3. ext_proc for `/v1/chat/completions` (profile `decode-filter` +
   `random-picker`) returns `x-gateway-destination-endpoint` set to a decode
   runner's Borneo IPv6 and port 9080 (`[2606:ec80:10:263d::20]:9080` or
   `[...::21]:9080`).
4. Setting a runner's slice `ready=false` removes it from routing.

## Required ECP changes

- ECP writes `ready/serving=true` unconditionally. The plugin honors the
  condition, so ECP must derive it from model readiness
  (`exos.google.com/model-ready`).
- ECP slices carry no role label. `decode-filter` accepts endpoints without a
  role, so prefill runners would be decode candidates. The plugin copies slice
  labels unchanged, so ECP must write `llm-d.ai/role` (`prefill` or `decode`)
  on each slice (simecp does this with `roleLabels=true`).
- The metrics extractor defaults to vLLM metric names. Set
  `defaultEngine: sglang` on `core-metrics-extractor`, or have ECP write
  `llm-d.ai/engine-type: sglang` on the slices.
- The EPP ServiceAccount needs `get/list/watch` on
  `discovery.k8s.io/endpointslices`, and the EPP runs with
  `--allow-experimental-plugins` (the plugin is Alpha).

Not covered: metrics scraping (needs mTLS or a proxy-pod metrics host), Envoy,
routing-proxy and P/D.
