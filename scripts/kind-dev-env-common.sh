# Sourced by scripts/kind-dev-env.sh and scripts/kind-dev-env-pools.sh.
# Owns the variable defaults, helpers, and stage functions that are identical
# across both topologies. Callers set MODEL_NAME (and any topology-specific
# variables that gate model derivation) before sourcing this file.

# ------------------------------------------------------------------------------
# Common variable defaults
# ------------------------------------------------------------------------------

: "${CLUSTER_NAME:=llm-d-router-dev}"
: "${GATEWAY_HOST_PORT:=30080}"
: "${IMAGE_REGISTRY:=ghcr.io/llm-d}"

export VLLM_SIMULATOR_TAG="${VLLM_SIMULATOR_TAG:-v0.9.0}"

# VLLM_IMAGE: the vLLM container image to deploy. Can be a simulator or real vLLM image
# (e.g., vllm/vllm-openai:v0.16.0 for production). Defaults to the simulator image.
export VLLM_IMAGE="${VLLM_IMAGE:-${IMAGE_REGISTRY}/llm-d-inference-sim:${VLLM_SIMULATOR_TAG}}"

export EPP_TAG="${EPP_TAG:-dev}"
EPP_IMAGE="${EPP_IMAGE:-${IMAGE_REGISTRY}/llm-d-router-endpoint-picker:${EPP_TAG}}"
export EPP_IMAGE

# Caller must set MODEL_NAME before sourcing.
export MODEL_FAMILY="${MODEL_NAME%%/*}"
export MODEL_ID="${MODEL_NAME##*/}"
export MODEL_NAME_SAFE=$(echo "${MODEL_ID}" | tr '[:upper:]' '[:lower:]' | tr ' /_.' '-')

export EPP_NAME="${EPP_NAME:-${MODEL_NAME_SAFE}-endpoint-picker}"

export SIDECAR_TAG="${SIDECAR_TAG:-dev}"
SIDECAR_IMAGE="${SIDECAR_IMAGE:-${IMAGE_REGISTRY}/llm-d-router-disagg-sidecar:${SIDECAR_TAG}}"
export SIDECAR_IMAGE

# CPU-only vLLM image that runs `vllm launch render` for the token-producer
# plugin's HTTP backend.
export VLLM_RENDER_IMAGE="${VLLM_RENDER_IMAGE:-vllm/vllm-openai-cpu:v0.21.0}"

export POOL_NAME="${POOL_NAME:-${MODEL_NAME_SAFE}-inference-pool}"

export PROM_ENABLED="${PROM_ENABLED:-false}"
: "${PROM_HOST_PORT:=30090}"

export KV_CACHE_ENABLED="${KV_CACHE_ENABLED:-false}"

export VLLM_REPLICA_COUNT_E="${VLLM_REPLICA_COUNT_E:-1}"
export VLLM_REPLICA_COUNT_P="${VLLM_REPLICA_COUNT_P:-1}"
export VLLM_REPLICA_COUNT_D="${VLLM_REPLICA_COUNT_D:-1}"
export VLLM_DATA_PARALLEL_SIZE="${VLLM_DATA_PARALLEL_SIZE:-1}"

# vLLM mode: echo for simulator, empty for real vLLM
export VLLM_SIM_MODE="${VLLM_SIM_MODE:-echo}"

# Empty by default — Kubernetes accepts empty label values, and the EPD patch
# uses this to optionally mark the unified pod's role.
export DECODE_ROLE="${DECODE_ROLE:-}"

export NAMESPACE="${NAMESPACE:-default}"
export METRICS_ENDPOINT_AUTH="${METRICS_ENDPOINT_AUTH:-false}"
export HF_TOKEN="${HF_TOKEN:-}"

# Extra vLLM args per pod type. Use --flag=value format.
# Example: VLLM_EXTRA_ARGS_D="--tensor-parallel-size=2"
export VLLM_EXTRA_ARGS_E="${VLLM_EXTRA_ARGS_E:-}"
export VLLM_EXTRA_ARGS_P="${VLLM_EXTRA_ARGS_P:-}"
export VLLM_EXTRA_ARGS_D="${VLLM_EXTRA_ARGS_D:-}"

# ------------------------------------------------------------------------------
# Image helpers — depend on CONTAINER_RUNTIME / PLATFORM_ARGS / SAVE_ARGS set
# by kind_pull_and_load_images.
# ------------------------------------------------------------------------------

pull_image() {
    local image="$1"
    if ! "${CONTAINER_RUNTIME}" image inspect "${image}" > /dev/null 2>&1; then
        echo "Image ${image} not found locally, pulling..."
        "${CONTAINER_RUNTIME}" pull ${PLATFORM_ARGS[@]+"${PLATFORM_ARGS[@]}"} "${image}"
    fi
}

load_image() {
    local image="$1"
    echo "Loading ${image} into kind cluster..."
    if [ "${CONTAINER_RUNTIME}" == "docker" ]; then
        # KIND's `kind load` uses `ctr import --all-platforms` internally, which
        # fails when only the target architecture's layers are locally cached
        # (e.g. after `docker pull --platform linux/amd64` of a multi-arch image).
        # Bypass this by piping directly to `ctr import` without --all-platforms.
        docker save "${image}" | \
            docker exec --privileged -i "${CLUSTER_NAME}-control-plane" \
            ctr --namespace=k8s.io images import --digests --snapshotter=overlayfs -
    else
        "${CONTAINER_RUNTIME}" save ${SAVE_ARGS[@]+"${SAVE_ARGS[@]}"} "${image}" | kind --name "${CLUSTER_NAME}" load image-archive /dev/stdin
    fi
}

# Retries the kustomize+apply pipeline up to 3 times with a 5-second backoff.
# etcd occasionally times out on large CRD sets (e.g. Istio); retrying is safe
# because --server-side --force-conflicts is idempotent.
apply_crds() {
    local kustomize_extra_flags="$1"
    local kustomize_dir="$2"
    local attempt max_attempts=3
    for attempt in $(seq 1 ${max_attempts}); do
        if kubectl kustomize ${kustomize_extra_flags} "${kustomize_dir}" \
               | kubectl --context ${KUBE_CONTEXT} apply --server-side --force-conflicts -f -; then
            return 0
        fi
        if [ "${attempt}" -lt "${max_attempts}" ]; then
            echo "CRD apply failed (attempt ${attempt}/${max_attempts}), retrying in 5s..." >&2
            sleep 5
        fi
    done
    echo "Error: CRD apply failed after ${max_attempts} attempts: ${kustomize_dir}" >&2
    return 1
}

# ------------------------------------------------------------------------------
# Stage functions
# ------------------------------------------------------------------------------

# Detect container runtime, enable nounset, verify required binaries, and
# validate inotify limits when Prometheus is requested.
kind_setup_checks() {
    if [ -z "${CONTAINER_RUNTIME:-}" ]; then
      if command -v docker &> /dev/null; then
        CONTAINER_RUNTIME="docker"
      elif command -v podman &> /dev/null; then
        CONTAINER_RUNTIME="podman"
      else
        echo "Neither docker nor podman could be found in PATH" >&2
        exit 1
      fi
    fi

    set -u

    local cmd
    for cmd in kind kubectl ${CONTAINER_RUNTIME}; do
        if ! command -v "$cmd" &> /dev/null; then
            echo "Error: $cmd is not installed or not in the PATH."
            exit 1
        fi
    done

    if [ "${PROM_ENABLED}" == "true" ]; then
      local inotify_instances
      inotify_instances=$(cat /proc/sys/fs/inotify/max_user_instances)
      if [ "${inotify_instances}" -lt 512 ]; then
        echo "Error: fs.inotify.max_user_instances is ${inotify_instances} (need >= 512) for Prometheus."
        echo ""
        echo "  sudo sysctl -w fs.inotify.max_user_instances=512"
        echo ""
        echo "To persist: echo 'fs.inotify.max_user_instances=512' | sudo tee /etc/sysctl.d/99-inotify.conf"
        exit 1
      fi
    fi
}

# TARGET_PORTS is substituted directly into the `targetPorts: ${TARGET_PORTS}`
# field in deploy/components/inference-gateway/.../inference-pool*.yaml. Each
# item must be indented with exactly 2 spaces to match the indentation of that
# field. If the field is ever reindented there, update the indentation here too.
kind_compute_target_ports() {
    local nl=$'\n'
    TARGET_PORTS="${nl}  - number: 8000"
    local i extra_port
    for ((i = 1; i < VLLM_DATA_PARALLEL_SIZE; ++i)); do
        extra_port=$((8000 + i))
        TARGET_PORTS="${TARGET_PORTS}${nl}  - number: ${extra_port}"
    done
    export TARGET_PORTS
}

# Create the kind cluster if it does not already exist. Reads the optional
# EXTRA_PORT_MAPPINGS env var (additional containerPort/hostPort entries) so
# topology-specific scripts can inject mappings without changing this signature.
# The Prometheus 30090 mapping is folded in automatically when PROM_ENABLED.
kind_create_cluster() {
    if kind get clusters 2>/dev/null | grep -q "^${CLUSTER_NAME}$"; then
        echo "Cluster '${CLUSTER_NAME}' already exists, re-using"
        return 0
    fi

    local prom_mapping=""
    if [ "${PROM_ENABLED}" == "true" ]; then
      prom_mapping="  - containerPort: 30090
    hostPort: ${PROM_HOST_PORT}
    protocol: TCP"
    fi

    local extra="${EXTRA_PORT_MAPPINGS:-}"
    if [ -n "${prom_mapping}" ]; then
      if [ -n "${extra}" ]; then
        extra="${extra}
${prom_mapping}"
      else
        extra="${prom_mapping}"
      fi
    fi

    kind create cluster --name "${CLUSTER_NAME}" --config - << EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
- role: control-plane
  # Pin to Kubernetes 1.31+ for Gateway API v1.5.1 compatibility
  # (requires isIP() CEL function and ValidatingAdmissionPolicy)
  image: kindest/node:v1.31.12
  extraPortMappings:
  - containerPort: 30080
    hostPort: ${GATEWAY_HOST_PORT}
    protocol: TCP
${extra}
EOF
}

# Set the kubectl context, apply the kind ARP hotfix, and wait for system pods.
kind_post_cluster_setup() {
    KUBE_CONTEXT="kind-${CLUSTER_NAME}"
    export KUBE_CONTEXT
    kubectl config set-context ${KUBE_CONTEXT} --namespace=default

    set -x

    local container_name="${CLUSTER_NAME}-control-plane"
    # Hotfix for https://github.com/kubernetes-sigs/kind/issues/3880
    ${CONTAINER_RUNTIME} exec ${container_name} /bin/bash -c "sysctl net.ipv4.conf.all.arp_ignore=0"

    kubectl --context ${KUBE_CONTEXT} -n kube-system wait --for=condition=Ready --all pods --timeout=300s

    echo "Waiting for local-path-storage pods to be created..."
    local deadline=$(( $(date +%s) + 120 ))
    until kubectl --context ${KUBE_CONTEXT} -n local-path-storage get pods -o name 2>/dev/null | grep -q pod/; do
      if (( $(date +%s) >= deadline )); then
        echo "ERROR: local-path-storage pods did not appear within 120s" >&2
        kubectl --context ${KUBE_CONTEXT} get namespaces >&2 || true
        kubectl --context ${KUBE_CONTEXT} -n local-path-storage get pods >&2 || true
        exit 1
      fi
      sleep 2
    done
    kubectl --context ${KUBE_CONTEXT} -n local-path-storage wait --for=condition=Ready --all pods --timeout=300s
}

# Variadic: pulls and loads each image into the kind cluster. Initialises
# CONTAINER_RUNTIME-specific PLATFORM_ARGS / SAVE_ARGS used by pull_image and
# load_image.
kind_pull_and_load_images() {
    local arch
    arch="$(uname -m)"
    case "${arch}" in
        x86_64) arch="amd64" ;;
        aarch64|arm64) arch="arm64" ;;
    esac

    PLATFORM_ARGS=()
    SAVE_ARGS=()
    if [ "${CONTAINER_RUNTIME}" == "docker" ]; then
        PLATFORM_ARGS=("--platform" "linux/${arch}")
    elif [ "${CONTAINER_RUNTIME}" == "podman" ]; then
        SAVE_ARGS=("--format=docker-archive")
    fi

    local image
    for image in "$@"; do
        pull_image "${image}"
        load_image "${image}"
    done
}

kind_apply_standard_crds() {
    apply_crds ""               deploy/components/crds-gateway-api
    apply_crds ""               deploy/components/crds-gie
    apply_crds ""               config/crd
    apply_crds "--enable-helm"  deploy/components/crds-istio
}

kind_wait_deployments_and_gateway() {
    kubectl --context ${KUBE_CONTEXT} -n llm-d-istio-system wait --for=condition=available --timeout=600s deployment --all
    kubectl --context ${KUBE_CONTEXT} -n default wait --for=condition=available --timeout=600s deployment --all
    kubectl --context ${KUBE_CONTEXT} wait gateway/inference-gateway --for=condition=Programmed --timeout=600s
}

kind_deploy_prometheus_if_enabled() {
    if [ "${PROM_ENABLED}" != "true" ]; then
        return 0
    fi

    echo "Deploying Prometheus monitoring stack..."

    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts 2>/dev/null || true
    helm repo update prometheus-community

    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring --create-namespace \
        --set grafana.enabled=false \
        --set alertmanager.enabled=false \
        --set kubeControllerManager.enabled=false \
        --set kubeEtcd.enabled=false \
        --set kubeProxy.enabled=false \
        --set kubeScheduler.enabled=false \
        --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.podMonitorSelectorNilUsesHelmValues=false \
        --set prometheus.prometheusSpec.resources.requests.memory=512Mi \
        --set prometheus.prometheusSpec.resources.limits.memory=1Gi \
        --set prometheus.service.type=NodePort \
        --set prometheus.service.nodePort=30090 \
        --kube-context ${KUBE_CONTEXT} \
        --wait --timeout 300s

    kubectl kustomize deploy/components/monitoring \
        | envsubst '${EPP_NAME} ${POOL_NAME}' \
        | kubectl --context ${KUBE_CONTEXT} apply -f -

    echo "Prometheus monitoring deployed."
}
