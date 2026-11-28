#!/usr/bin/env python3
# Copyright 2026 The llm-d Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Prepare issue 913 A-C inference-perf configurations without starting a load test."""

import argparse
import json
import math
from pathlib import Path
from urllib.parse import urlsplit


SCENARIOS = {
    "A": (6, 1000, 1000, 30, 9),
    "B": (6, 1000, 1000, 3000, 900),
    "C": (150, 5, 6000, 1200, 360),
}
STRATEGIES = {"k8s", "load-prefix", "pred-before", "pred-after"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("schema", "registry"), required=True)
    parser.add_argument(
        "--target", action="append", required=True, metavar="STRATEGY=URL"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--qps", type=float, nargs="+", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--max-concurrency-per-worker", type=int, required=True)
    parser.add_argument(
        "--clip-sigma",
        type=float,
        required=True,
        help="Explicit upper length bound: mean + N * std_dev",
    )
    parser.add_argument("--seed", type=int, default=913)
    parser.add_argument("--duration", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    targets = {}
    for target in args.target:
        name, separator, url = target.partition("=")
        try:
            endpoint = urlsplit(url)
        except ValueError:
            parser.error("--target contains an invalid URL")
        if (
            not separator
            or name not in STRATEGIES
            or name in targets
            or endpoint.scheme not in ("http", "https")
            or not endpoint.hostname
        ):
            parser.error(
                "--target must uniquely specify each of k8s, load-prefix, pred-before and pred-after"
            )
        targets[name] = url
    if set(targets) != STRATEGIES:
        parser.error("all four --target strategies are required")
    if not args.tokenizer_path.is_dir():
        parser.error("--tokenizer-path must be an existing local directory")
    if not all(
        math.isfinite(qps) and qps > 0 for qps in args.qps
    ) or args.qps != sorted(set(args.qps)):
        parser.error("--qps must be finite, positive and strictly increasing")
    if min(args.workers, args.max_concurrency_per_worker, args.duration) < 1:
        parser.error("worker, concurrency and duration values must be positive")
    stage_sizes = [qps * args.duration for qps in args.qps]
    if any(not math.isfinite(size) or size < 1 for size in stage_sizes):
        parser.error("each stage must schedule at least one request")
    if not math.isfinite(args.clip_sigma) or args.clip_sigma <= 0:
        parser.error("--clip-sigma must be finite and positive")

    def distribution(mean, std_dev):
        return {
            "type": "normal",
            "mean": mean,
            "std_dev": std_dev,
            "min": 1,
            "max": math.ceil(mean + args.clip_sigma * std_dev),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for scenario, (groups, users, system, question, std_dev) in SCENARIOS.items():
        for strategy, url in sorted(targets.items()):
            name = f"{args.phase}-{scenario}-{strategy}"
            config = {
                "api": {"type": "completion", "streaming": True},
                "server": {
                    "type": "vllm",
                    "model_name": args.model,
                    "base_url": url,
                    "ignore_eos": True,
                },
                "tokenizer": {
                    "pretrained_model_name_or_path": str(args.tokenizer_path.resolve()),
                    "trust_remote_code": False,
                },
                "load": {
                    "type": "constant",
                    "base_seed": args.seed,
                    "num_workers": args.workers,
                    "worker_max_concurrency": args.max_concurrency_per_worker,
                    "stages": [
                        {"rate": qps, "duration": args.duration} for qps in args.qps
                    ],
                },
                "data": {
                    "type": "shared_prefix",
                    "shared_prefix": {
                        "num_groups": groups,
                        "num_prompts_per_group": users,
                        "system_prompt_len": system,
                        "question_len": distribution(question, std_dev),
                        "output_len": distribution(1000, 300),
                        "enable_multi_turn_chat": True,
                        "seed": args.seed,
                    },
                },
                "report": {
                    "request_lifecycle": {
                        "summary": True,
                        "per_stage": True,
                        "per_request": True,
                    }
                },
                "storage": {"local_storage": {"path": f"reports/{name}"}},
            }
            filename = name + ".json"
            (args.output_dir / filename).write_text(json.dumps(config, indent=2) + "\n")
            files.append(filename)
    manifest = {
        "status": "prepared-not-run",
        "inference_perf_version": "v0.6.0",
        "inference_perf_commit": "e28d9a0bf5cefa743910b73057b3b686f0a94b98",
        "inference_perf_source_tree": "84ff8848c53a22f0c30636466feeb904a881e262",
        "issue": "https://github.com/llm-d/llm-d-router/issues/913",
        "phase": args.phase,
        "length_bounds": {"minimum": 1, "upper_sigma": args.clip_sigma},
        "nominal_repeated_session_requests": {
            scenario: [max(0, int(size) - groups * users) for size in stage_sizes]
            for scenario, (groups, users, *_lengths) in SCENARIOS.items()
        },
        "configs": files,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(
        f"Prepared {len(files)} configurations; no servers or load generators started."
    )


if __name__ == "__main__":
    main()
