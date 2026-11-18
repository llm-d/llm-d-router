# Pod-local vLLM frontend

The Service exposes the coordinator on port 8080. With `server.vllm_frontend: true` in the mounted coordinator configuration, the coordinator forwards POST requests to `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/tokenize`, and `/detokenize` to the Rust frontend on loopback port 8000. The frontend handles HTTP parsing, tokenization, and response formatting. For generation it calls `vllm.Inference/GenerateStream` with token IDs on the same coordinator's loopback port 8081.

Public vllm.Inference Generate and GenerateStream requests go to the Rust frontend on loopback port 50051. Both text and token-ID prompts use the frontend adapter and its supported generation options. Other routes use their ordinary coordinator handlers. The example serves cleartext HTTP and gRPC on the public port. Terminate external TLS at the cluster ingress.

## Run

Use Kubernetes 1.29 or newer with native sidecars enabled, an inference-aware Gateway and one ready aggregated vLLM Rust worker. Serve Qwen/Qwen3-0.6B at revision `c1899de289a04d12100db370d81485cdf75e47ca` with a context limit of 512, native gRPC on port 50051 and HTTP metrics on port 8000. Label the worker pod `app=generation-worker`. The frontend needs access to the pinned Hugging Face config and tokenizer or a populated local cache.

1. Build the two images from the llm-d-router root. Make them available on the cluster nodes, or substitute their registry references in the deployment.

   ```sh
   docker build -f Dockerfile.coordinator -t coordinator-frontend:local .
   docker build -t grpc-generation-frontend:local ../vllm/examples/rust_grpc_frontend
   ```

2. Set the gateway address in `coordinator.yaml` to the inference gateway Service. Kustomize mounts this configuration in the coordinator container. For the gateway chart, install the explicit native parser configuration and adjust the Gateway name if needed.

   ```sh
   example=config/examples/frontend-proxy
   chart=config/charts/llm-d-router-gateway
   helm dependency build "$chart"
   helm upgrade --install native-generation "$chart" -f "$example/epp-values.yaml"
   kubectl apply -k "$example"
   kubectl port-forward svc/frontend-proxy 8080:8080
   ```

3. Send a request through the public port.

   ```sh
   curl -NsS http://localhost:8080/v1/chat/completions \
     -H 'Content-Type: application/json' \
     -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"Count from one."}],"max_tokens":8,"temperature":0,"stream":true}'
   ```

## Verify

Run `go test -race ./internal/frontendproxy ./cmd/coordinator ./pkg/coordinator/server` for protocol and dispatch checks. With the public service and direct worker ports forwarded locally, run the following check against an otherwise idle worker. Install grpcurl and use the inference.proto from the frontend builder revision. For Chat parity use the same model identity, Qwen reasoning parser, tokenizer and generation defaults on both servers.

```sh
python3 config/examples/frontend-proxy/verify.py --public http://localhost:8080 \
  --worker http://localhost:18000 --worker-grpc localhost:15051 \
  --proto-dir /path/to/vllm-at-21d7d929/rust/proto
```

If the worker HTTP server lacks Responses, run a second copy of the pinned Rust frontend with `--generation-url` pointing directly to the worker and pass its HTTP origin as `--responses-reference`.

The check compares direct and routed results and measures worker aborts for a disconnected client and a stop string spanning two token deltas. Discovery follows the lower router's normal route and requires that stack to support HTTP model discovery. [RFC](https://github.com/llm-d/llm-d-router/issues/3617) · [Etai's proposal](https://github.com/llm-d/llm-d-router/pull/3992#issuecomment-6831885658)
