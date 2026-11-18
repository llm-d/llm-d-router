"""Compare the public ingress with an idle, directly accessible Qwen worker."""

import argparse
import json
import subprocess
import time
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def post(base, path, body):
    return urlopen(
        Request(
            base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        ),
        timeout=30,
    )


def events(response):
    for line in response:
        if line.startswith(b"data:"):
            data = line[5:].strip()
            yield None if data == b"[DONE]" else json.loads(data)


def result(base, path, body):
    with post(base, path, body) as response:
        if body["stream"]:
            frames = list(events(response))
        else:
            frames = [json.load(response)]
    if path == "/v1/responses":
        if body["stream"]:
            frame = next(
                f["response"]
                for f in frames
                if f and f["type"] in ("response.completed", "response.incomplete")
            )
            deltas = "".join(
                f["delta"]
                for f in frames
                if f
                and f["type"]
                in ("response.output_text.delta", "response.reasoning_text.delta")
            )
        else:
            frame = frames[0]
        text = "".join(p["text"] for item in frame["output"] for p in item["content"])
        if body["stream"]:
            assert deltas == text
        counts = frame["usage"]
        return (
            text,
            frame["status"],
            frame["incomplete_details"],
            counts["input_tokens"],
            counts["output_tokens"],
        )
    if body["stream"]:
        assert frames.pop() is None
    choices = [c for f in frames for c in f["choices"]]
    messages = [c.get("delta", c.get("message", {})) for c in choices]
    text = "".join(
        c.get("text") or m.get("content") or "" for c, m in zip(choices, messages)
    )
    reasoning = "".join(m.get("reasoning") or "" for m in messages)
    usage = next(f["usage"] for f in reversed(frames) if f.get("usage"))
    return (
        text,
        reasoning,
        choices[-1]["finish_reason"],
        usage["prompt_tokens"],
        usage["completion_tokens"],
    )


def rpc(host, method, args, prompt_input):
    req = {
        "model": args.model,
        **prompt_input,
        "temperature": 0,
        "stopping": {"maxNewTokens": 4},
        "response": {
            "outputText": False,
            "outputTokenIds": True,
            "promptTokenIds": True,
        },
    }
    raw = subprocess.check_output(
        [
            "grpcurl",
            "-plaintext",
            "-max-time",
            "30",
            "-import-path",
            args.proto_dir,
            "-proto",
            "inference.proto",
            "-d",
            json.dumps(req),
            host,
            "vllm.Inference/" + method,
        ],
        text=True,
    )
    prompt, tokens, finish = [], [], None
    while raw.strip():
        frame, end = json.JSONDecoder().raw_decode(raw.lstrip())
        raw = raw.lstrip()[end:]
        prompt.extend(frame.get("promptInfo", {}).get("tokenIds", []))
        out = frame.get("outputs", {})
        tokens.extend(out.get("tokenIds", []))
        finish = out.get("finishInfo", {}).get("finishReason", finish)
    assert prompt and len(tokens) == 4 and finish == "LENGTH"
    if "tokenIds" in prompt_input:
        assert prompt == prompt_input["tokenIds"]["ids"]
    return prompt, tokens


def abort_count(worker):
    with urlopen(worker + "/metrics", timeout=5) as res:
        return sum(
            float(line.split()[-1])
            for line in res
            if line.startswith(b"vllm:request_success_total{")
            and b'finished_reason="abort"' in line
        )


def verify_release(args, params):
    path = "/v1/completions"
    with post(args.worker, path, dict(params, max_tokens=4)) as response:
        pieces = [
            c["text"]
            for f in events(response)
            if f
            for c in f["choices"]
            if c.get("text")
        ]
    assert len(pieces) >= 2
    stop = pieces[0][-1] + pieces[1]
    for case in ("disconnect", "split stop"):
        before = abort_count(args.worker)
        if case == "disconnect":
            with post(args.public, path, params) as response:
                next(
                    c["text"]
                    for f in events(response)
                    if f
                    for c in f["choices"]
                    if c.get("text")
                )
        else:
            got = result(args.public, path, dict(params, stop=[stop]))
            assert got[0] == pieces[0][:-1] and got[2] == "stop", got
        deadline = time.monotonic() + 15
        while abort_count(args.worker) == before and time.monotonic() < deadline:
            time.sleep(0.2)
        assert abort_count(args.worker) == before + 1, case
        print(case, "released one worker request")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("public", "worker", "worker-grpc", "proto-dir"):
        parser.add_argument("--" + flag, required=True)
    parser.add_argument(
        "--responses-reference",
        help="HTTP frontend connected directly to the worker when its HTTP API lacks Responses",
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()
    prompt = "Count from one."
    messages = [{"role": "user", "content": prompt}]
    for path, body in [
        ("/tokenize", {"messages": messages}),
        ("/detokenize", {"tokens": [42, 314, 2718]}),
    ]:
        with (
            post(args.public, path, dict(body, model=args.model)) as ingress,
            post(args.worker, path, dict(body, model=args.model)) as direct,
        ):
            assert json.load(ingress) == json.load(direct)
    with urlopen(args.public + "/v1/models", timeout=5) as response:
        assert args.model in [m["id"] for m in json.load(response)["data"]]
    for path, body in [
        ("/v1/completions", {"prompt": prompt}),
        ("/v1/chat/completions", {"messages": messages}),
        ("/v1/responses", {"input": prompt}),
    ]:
        for stream in (False, True):
            params = dict(body, temperature=0, model=args.model, stream=stream)
            params["max_output_tokens" if "input" in body else "max_tokens"] = 8
            if stream and "input" not in body:
                params["stream_options"] = {"include_usage": True}
            got = result(args.public, path, params)
            reference = args.worker
            if path == "/v1/responses" and args.responses_reference:
                reference = args.responses_reference
            assert got == result(reference, path, params) and got[-1] == 8, got
            print(path, "stream", stream, got)
    for method in ("Generate", "GenerateStream"):
        for prompt_input in ({"text": prompt}, {"tokenIds": {"ids": [42, 314, 2718]}}):
            assert rpc(urlsplit(args.public).netloc, method, args, prompt_input) == rpc(
                args.worker_grpc, method, args, prompt_input
            )
            print(method, next(iter(prompt_input)), "matched direct worker")
    verify_release(
        args,
        dict(
            prompt=prompt,
            temperature=0,
            model=args.model,
            max_tokens=256,
            ignore_eos=True,
            stream=True,
            stream_options={"include_usage": True},
        ),
    )


if __name__ == "__main__":
    main()
