# GuideLLM workloads

`run.sh` uses the GuideLLM 0.7 `run` CLI directly. Set `TARGET` to the inference
Gateway base URL. The default model is `Qwen/Qwen3-0.6B`, and `OUTPUT_DIR`
defaults to `./guidellm-results`.

```sh
TARGET=http://GATEWAY_ADDRESS ./run.sh fixed non-streaming
TARGET=http://GATEWAY_ADDRESS ./run.sh all non-streaming
TARGET=http://GATEWAY_ADDRESS ./run.sh all streaming
```

Available scenarios are:

| Name | Profile | Input/output tokens | Load values |
|---|---|---:|---|
| `fixed` | Poisson | 200/100 | 400 requests/s |
| `rate` | Poisson | 200/100 | 25, 100, 400, 800 requests/s |
| `short` | Poisson | 200/1 | 100 requests/s |
| `long-input` | Poisson | 8000/100 | 20 requests/s |
| `long-output` | Poisson | 200/1000 | 10 requests/s |
| `concurrency` | Concurrent | 200/100 | 1, 32, 128 streams |

Every load stage runs for 300 seconds. The maximum Poisson concurrency is 512,
the synthetic-data seed is 20260919, and detailed request retention is capped
at 100 requests per status group. Run the script from a load-generator host or
pod with enough file descriptors and network capacity for the 800 requests/s
and concurrency-128 stages.
