# Testing

Automated P/D coverage of the sidecar lives in `test/e2e` ("Running a PD configuration with nixlv2 connector and metrics validation") and `pkg/sidecar/proxy`. This directory holds manifests and steps for manual deployment.

## Testing the NIXLConnector

The [nixl](config/overlays/llmd/nixl) directory contains the configuration files deploying
a simple 1P1D sample application using the NIXL connector.

To deploy this application in the `eval` cluster, run this command:

```
$ kubectl kustomize test/sidecar/config/overlays/llmd/nixl | oc apply -f -
```

Wait a bit (up to 10mn) for the pods to be running.

### Sending a request to the decode sidecar

Port-forward the decode sidecar port on your local machine:

```
$ oc port-forward pod/qwen-decoder 8000:8000
```

Send a request:

```
$ curl http://localhost:8000/v1/completions \
      -H "Content-Type: application/json" \
      -H "x-llm-d-prefiller-host-port: qwen-prefiller:8000" \
      -d '{"model": "Qwen/Qwen2-0.5B", "prompt": "Question: Greta worked 40 hours and was paid $12 per hour. Her friend Lisa earned $15 per hour at her job. How many hours would Lisa have to work to equal Gretas earnings for 40 hours?", "max_tokens": 200 }'
```

Observe the decoder logs:

```
$ oc logs qwen-decoder
...
NFO 05-06 01:42:03 [logger.py:39] Received request cmpl-4fa3a7af-2a1b-11f0-854f-0a580a800726-0: prompt: 'Question: Greta worked 40 hours and was paid $12 per hour. Her friend Lisa earned $15 per hour at her job. How many hours would Lisa have to work to equal Gretas earnings for 40 hours?', params: SamplingParams(n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0, temperature=1.0, top_p=1.0, top_k=-1, min_p=0.0, seed=None, stop=[], stop_token_ids=[], bad_words=[], include_stop_str_in_output=False, ignore_eos=False, max_tokens=200, min_tokens=0, logprobs=None, prompt_logprobs=None, skip_special_tokens=True, spaces_between_special_tokens=True, truncate_prompt_tokens=None, guided_decoding=None, extra_args=None), prompt_token_ids: [14582, 25, 479, 65698, 6439, 220, 19, 15, 4115, 323, 572, 7171, 400, 16, 17, 817, 6460, 13, 6252, 4238, 28556, 15303, 400, 16, 20, 817, 6460, 518, 1059, 2618, 13, 2585, 1657, 4115, 1035, 28556, 614, 311, 975, 311, 6144, 87552, 300, 23681, 369, 220, 19, 15, 4115, 30], lora_request: None, prompt_adapter_request: None.
INFO 05-06 01:42:03 [async_llm.py:255] Added request cmpl-4fa3a7af-2a1b-11f0-854f-0a580a800726-0.
DEBUG 05-06 01:42:03 [core.py:431] EngineCore loop active.
DEBUG 05-06 01:42:03 [nixl_connector.py:545] start_load_kv for request cmpl-4fa3a7af-2a1b-11f0-854f-0a580a800726-0 from remote engine 2faadfa5-23cf-4e58-81c3-4cfb89887471. Num local_block_ids: 3. Num remote_block_ids: 3.
DEBUG 05-06 01:42:03 [nixl_connector.py:449] Rank 0, get_finished: 0 requests done sending and 1 requests done recving
DEBUG 05-06 01:42:03 [scheduler.py:862] Finished recving KV transfer for request cmpl-4fa3a7af-2a1b-11f0-854f-0a580a800726-0
INFO 05-06 01:42:03 [loggers.py:116] Engine 000: Avg prompt throughput: 5.0 tokens/s, Avg generation throughput: 1.6 tokens/s, Running: 1 reqs, Waiting: 0 reqs, GPU KV cache usage: 0.0%, Prefix cache hit rate: 50.0%
DEBUG 05-06 01:42:05 [core.py:425] EngineCore waiting for work.
INFO:     ::1:0 - "POST /v1/completions HTTP/1.1" 200 OK
```
