#!/usr/bin/env bash
# Infrastructure for the GKE PoC: dual-stack VPC, dual-stack GKE cluster,
# GPU VMs running vLLM outside the cluster (stand-ins for Borneo linked
# runners), Cloud NAT, firewall, control-node annotations.
#
# Usage: infra.sh up | annotate | status | down
set -euo pipefail

PROJECT="${PROJECT:-xiyue-gke-dev-new}"
REGION="${REGION:-europe-west4}"
ZONE="${ZONE:-europe-west4-a}"
# Runner VMs may sit in another zone of the region: the subnet is regional.
RUNNER_ZONE="${RUNNER_ZONE:-${ZONE}}"
PREFIX="${PREFIX:-epp-poc}"
NETWORK="${PREFIX}-net"
SUBNET="${PREFIX}-subnet"
SUBNET_RANGE="${SUBNET_RANGE:-10.10.0.0/20}"
CLUSTER="${PREFIX}"
RUNNERS="${RUNNERS:-runner-0 runner-1}"
RUNNER_TAG="${PREFIX}-runner"
RUNNER_MACHINE="${RUNNER_MACHINE:-g2-standard-8}"
RUNNER_GPU="${RUNNER_GPU:-type=nvidia-l4,count=1}"
RUNNER_IMAGE_FAMILY="${RUNNER_IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:latest}"
RUNNER_PORT=9080
HERE="$(cd "$(dirname "$0")" && pwd)"

g() { gcloud --project "${PROJECT}" "$@"; }
exists() { "$@" >/dev/null 2>&1; }

up() {
  # Dual-stack VPC: internal (ULA) IPv6 so pods and VMs can use IPv6 later.
  exists g compute networks describe "${NETWORK}" ||
    g compute networks create "${NETWORK}" --subnet-mode=custom --enable-ula-internal-ipv6
  exists g compute networks subnets describe "${SUBNET}" --region "${REGION}" ||
    g compute networks subnets create "${SUBNET}" --network "${NETWORK}" --region "${REGION}" \
      --range "${SUBNET_RANGE}" --stack-type IPV4_IPV6 --ipv6-access-type INTERNAL

  # Runners have no external IP: NAT for Docker Hub / Hugging Face pulls.
  exists g compute routers describe "${PREFIX}-router" --region "${REGION}" ||
    g compute routers create "${PREFIX}-router" --network "${NETWORK}" --region "${REGION}"
  exists g compute routers nats describe "${PREFIX}-nat" --router "${PREFIX}-router" --region "${REGION}" ||
    g compute routers nats create "${PREFIX}-nat" --router "${PREFIX}-router" --region "${REGION}" \
      --auto-allocate-nat-external-ips --nat-all-subnet-ip-ranges

  # Dual-stack GKE requires Dataplane V2.
  exists g container clusters describe "${CLUSTER}" --zone "${ZONE}" ||
    g container clusters create "${CLUSTER}" --zone "${ZONE}" \
      --network "${NETWORK}" --subnetwork "${SUBNET}" \
      --enable-ip-alias --enable-dataplane-v2 \
      --stack-type ipv4-ipv6 \
      --num-nodes 2 --machine-type e2-standard-4 --release-channel regular
  g container clusters get-credentials "${CLUSTER}" --zone "${ZONE}"

  local pod_cidr
  pod_cidr="$(g container clusters describe "${CLUSTER}" --zone "${ZONE}" --format='value(clusterIpv4Cidr)')"
  exists g compute firewall-rules describe "${PREFIX}-to-runner" ||
    g compute firewall-rules create "${PREFIX}-to-runner" --network "${NETWORK}" \
      --allow "tcp:${RUNNER_PORT}" --target-tags "${RUNNER_TAG}" \
      --source-ranges "${pod_cidr},${SUBNET_RANGE}"
  exists g compute firewall-rules describe "${PREFIX}-iap-ssh" ||
    g compute firewall-rules create "${PREFIX}-iap-ssh" --network "${NETWORK}" \
      --allow tcp:22 --target-tags "${RUNNER_TAG}" --source-ranges 35.235.240.0/20

  for r in ${RUNNERS}; do
    exists g compute instances describe "${r}" --zone "${RUNNER_ZONE}" ||
      g compute instances create "${r}" --zone "${RUNNER_ZONE}" \
        --machine-type "${RUNNER_MACHINE}" --accelerator "${RUNNER_GPU}" \
        --maintenance-policy TERMINATE --provisioning-model "${RUNNER_PROVISIONING:-STANDARD}" ${RUNNER_PROVISIONING_FLAGS:-} \
        --subnet "${SUBNET}" --stack-type IPV4_IPV6 --no-address --tags "${RUNNER_TAG}" \
        --image-project deeplearning-platform-release --image-family "${RUNNER_IMAGE_FAMILY}" \
        --boot-disk-size 200GB \
        --metadata "install-nvidia-driver=True,model=${MODEL},vllm-image=${VLLM_IMAGE},runner-port=${RUNNER_PORT}" \
        --metadata-from-file "startup-script=${HERE}/vllm-startup.sh"
  done
}

runner_ip() { g compute instances describe "$1" --zone "${RUNNER_ZONE}" --format='value(networkInterfaces[0].networkIP)'; }

# annotate plays the linked-runner controller: one
# linked-runner.runner-instance/<runner> annotation per VM on the control node.
annotate() {
  local node
  node="$(kubectl get nodes -l exos-poc/control-node=true -o name | head -1)"
  if [[ -z "${node}" ]]; then
    node="$(kubectl get nodes -o name | head -1)"
    kubectl label "${node}" exos-poc/control-node=true --overwrite
  fi
  local i=0
  for r in ${RUNNERS}; do
    local ip
    ip="$(runner_ip "${r}")"
    kubectl annotate "${node}" --overwrite \
      "linked-runner.runner-instance/${r}={\"NAME\":\"${r}\",\"STATUS\":\"RUNNING\",\"IP_ADDRESS\":\"${ip}\",\"PROVIDER_ID\":\"provider://bns/${ZONE}/borg/${ZONE}/bns/poc/${PREFIX}/${i}\"}"
    i=$((i + 1))
  done
  echo "control node: ${node}"
}

status() {
  g compute instances list --filter="tags.items=${RUNNER_TAG}" \
    --format="table(name,status,networkInterfaces[0].networkIP,networkInterfaces[0].ipv6Address)"
  kubectl get nodes -l exos-poc/control-node=true -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{range .metadata.annotations}{end}{end}'
  kubectl get endpointslice -n default -l kubernetes.io/service-name=glm52-server -o wide || true
}

down() {
  for r in ${RUNNERS}; do g compute instances delete "${r}" --zone "${RUNNER_ZONE}" --quiet || true; done
  g container clusters delete "${CLUSTER}" --zone "${ZONE}" --quiet || true
  for fw in "${PREFIX}-to-runner" "${PREFIX}-iap-ssh"; do g compute firewall-rules delete "${fw}" --quiet || true; done
  g compute routers delete "${PREFIX}-router" --region "${REGION}" --quiet || true
  g compute networks subnets delete "${SUBNET}" --region "${REGION}" --quiet || true
  g compute networks delete "${NETWORK}" --quiet || true
}

case "${1:-}" in
  up) up ;;
  annotate) annotate ;;
  status) status ;;
  down) down ;;
  *) echo "usage: $0 up|annotate|status|down" >&2; exit 2 ;;
esac
