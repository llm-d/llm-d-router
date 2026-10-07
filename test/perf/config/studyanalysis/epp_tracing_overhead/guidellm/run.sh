#!/usr/bin/env bash

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

set -euo pipefail

scenario=${1:-}
response_mode=${2:-non-streaming}

if [[ -z "$scenario" ]]; then
  echo "usage: TARGET=http://host $0 SCENARIO [non-streaming|streaming]" >&2
  exit 2
fi

: "${TARGET:?set TARGET to the inference Gateway base URL}"

study_model=${MODEL:-Qwen/Qwen3-0.6B}
study_output_dir=${OUTPUT_DIR:-./guidellm-results}
mkdir -p "$study_output_dir"

case "$response_mode" in
  non-streaming) stream=false ;;
  streaming) stream=true ;;
  *)
    echo "response mode must be non-streaming or streaming" >&2
    exit 2
    ;;
esac

run_rate() {
  local name=$1
  local prompt_tokens=$2
  local output_tokens=$3
  local rates=$4

  guidellm run \
    --backend "{\"kind\":\"openai_http\",\"target\":\"$TARGET\",\"model\":\"$study_model\",\"stream\":$stream,\"timeout\":600,\"validate_backend\":false}" \
    --profile '{"kind":"poisson","max_concurrency":512}' \
    --override profile.rate "$rates" \
    --metrics '{"kind":"generative","sample_size":100}' \
    --seed '{"kind":"static","value":20260919}' \
    --data "{\"kind\":\"synthetic_text\",\"prompt_tokens\":$prompt_tokens,\"output_tokens\":$output_tokens}" \
    --tokenizer "{\"kind\":\"huggingface_auto\",\"model\":\"$study_model\"}" \
    --constraint '{"kind":"max_duration","seconds":300}' \
    --output "{\"kind\":\"json\",\"path\":\"$study_output_dir/$name-$response_mode.json\"}" \
    --disable-console-interactive
}

run_concurrency() {
  guidellm run \
    --backend "{\"kind\":\"openai_http\",\"target\":\"$TARGET\",\"model\":\"$study_model\",\"stream\":$stream,\"timeout\":600,\"validate_backend\":false}" \
    --profile '{"kind":"concurrent"}' \
    --override profile.streams 1,32,128 \
    --metrics '{"kind":"generative","sample_size":100}' \
    --seed '{"kind":"static","value":20260919}' \
    --data '{"kind":"synthetic_text","prompt_tokens":200,"output_tokens":100}' \
    --tokenizer "{\"kind\":\"huggingface_auto\",\"model\":\"$study_model\"}" \
    --constraint '{"kind":"max_duration","seconds":300}' \
    --output "{\"kind\":\"json\",\"path\":\"$study_output_dir/concurrency-200-100-$response_mode.json\"}" \
    --disable-console-interactive
}

run_named_scenario() {
  case "$1" in
    fixed) run_rate fixed-400-200-100 200 100 400 ;;
    rate) run_rate rate-200-100 200 100 25,100,400,800 ;;
    short) run_rate rate-200-1 200 1 100 ;;
    long-input) run_rate rate-8000-100 8000 100 20 ;;
    long-output) run_rate rate-200-1000 200 1000 10 ;;
    concurrency) run_concurrency ;;
    *)
      echo "unknown scenario: $1" >&2
      exit 2
      ;;
  esac
}

if [[ "$scenario" == all ]]; then
  for item in rate short long-input long-output concurrency; do
    run_named_scenario "$item"
  done
else
  run_named_scenario "$scenario"
fi
