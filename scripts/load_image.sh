#!/bin/bash

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


# Use the CONTAINER_RUNTIME from the environment, or default to docker if it's not set.
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-docker}"
echo "Using container tool: ${CONTAINER_RUNTIME}"


LINUX_ARCH="$(uname -m)"
case "${LINUX_ARCH}" in
    x86_64) LINUX_ARCH="amd64" ;;
    aarch64|arm64) LINUX_ARCH="arm64" ;;
esac

PLATFORM_ARGS=()
SAVE_ARGS=()
if [ "${CONTAINER_RUNTIME}" == "docker" ]; then
    PLATFORM_ARGS=("--platform" "linux/${LINUX_ARCH}")
elif [ "${CONTAINER_RUNTIME}" == "podman" ]; then
    SAVE_ARGS=("--format=docker-archive")
fi

for IMAGE in $@; do
    # KIND's `kind load` uses `ctr import --all-platforms` internally, which
    # fails when only the target architecture's layers are locally cached
    # (e.g. after `docker pull --platform linux/amd64` of a multi-arch image).
    echo "Loading $IMAGE to the ${CLUSTER_NAME} kind cluster"
    "${CONTAINER_RUNTIME}" save ${PLATFORM_ARGS[@]+"${PLATFORM_ARGS[@]}"} ${SAVE_ARGS[@]+"${SAVE_ARGS[@]}"} "${IMAGE}" | kind --name "${CLUSTER_NAME}" load image-archive /dev/stdin
done
