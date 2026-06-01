#!/bin/bash

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
