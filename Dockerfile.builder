FROM golang:1.27.3

RUN mkdir /app
WORKDIR /app

ARG TYPOS_VERSION=v1.34.0
ARG KIND_VERSION=v0.33.0
ARG GOLANGCI_LINT_VERSION=v2.14.0
ARG KUBECTL_VERSION=v1.35.9
ARG KUSTOMIZE_VERSION=v5.8.1
ARG DOCKER_VERSION=29.8.1
ARG DOCKER_BUILDX_VERSION=v0.37.1
ARG ENVTEST_VERSION=release-0.25
ARG ENVTEST_K8S_VERSION=1.36.2
ARG GOVULNCHECK_VERSION=v1.8.0

RUN apt-get update && apt-get install -y podman && apt-get clean all

# Install docker CLI and buildx plugin
RUN ARCH=$(uname -m) && \
    GOARCH=$(echo ${ARCH} | sed 's/x86_64/amd64/; s/aarch64/arm64/') && \
    curl -sSfL https://download.docker.com/linux/static/stable/${ARCH}/docker-${DOCKER_VERSION}.tgz | \
    tar -xz --strip-components=1 -C /usr/local/bin docker/docker && \
    mkdir -p /usr/local/lib/docker/cli-plugins && \
    curl -sSfL -o /usr/local/lib/docker/cli-plugins/docker-buildx \
      "https://github.com/docker/buildx/releases/download/${DOCKER_BUILDX_VERSION}/buildx-${DOCKER_BUILDX_VERSION}.linux-${GOARCH}" && \
    chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx

# Install golangci-lint
RUN GOARCH=$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/') && \
    curl -sSfL https://github.com/golangci/golangci-lint/releases/download/${GOLANGCI_LINT_VERSION}/golangci-lint-${GOLANGCI_LINT_VERSION#v}-linux-${GOARCH}.tar.gz | \
    tar -xz --strip-components=1 -C /usr/local/bin golangci-lint-${GOLANGCI_LINT_VERSION#v}-linux-${GOARCH}/golangci-lint

# Install kubectl
RUN GOARCH=$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/') && \
    curl -sSfL -o /usr/local/bin/kubectl "https://dl.k8s.io/release/${KUBECTL_VERSION}/bin/linux/${GOARCH}/kubectl" && \
    chmod +x /usr/local/bin/kubectl

# Install kustomize
RUN GOARCH=$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/') && \
    curl -sSfL https://github.com/kubernetes-sigs/kustomize/releases/download/kustomize/${KUSTOMIZE_VERSION}/kustomize_${KUSTOMIZE_VERSION}_linux_${GOARCH}.tar.gz | \
    tar -xz -C /usr/local/bin

# Install kind
RUN GOARCH=$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/') && \
    curl -sSfL -o /usr/local/bin/kind https://github.com/kubernetes-sigs/kind/releases/download/${KIND_VERSION}/kind-linux-${GOARCH} && \
    chmod +x /usr/local/bin/kind

# Install typos
RUN curl -sSfL https://github.com/crate-ci/typos/releases/download/${TYPOS_VERSION}/typos-${TYPOS_VERSION}-$(uname -m)-unknown-linux-musl.tar.gz | tar -xz -C /usr/local/bin

# Install setup-envtest and pre-fetch kube test binaries so hermetic integration
# tests don't need network at runtime. ENVTEST_ASSETS_DIR is exported so the
# Makefile target can resolve KUBEBUILDER_ASSETS without re-running setup-envtest.
ENV ENVTEST_ASSETS_DIR=/usr/local/kubebuilder
RUN GOBIN=/usr/local/bin go install sigs.k8s.io/controller-runtime/tools/setup-envtest@${ENVTEST_VERSION} && \
    setup-envtest use ${ENVTEST_K8S_VERSION} --bin-dir ${ENVTEST_ASSETS_DIR} && \
    chmod -R a+rx ${ENVTEST_ASSETS_DIR}
ENV ENVTEST_K8S_VERSION=${ENVTEST_K8S_VERSION}

# Install govulncheck. Build-time install ensures runtime invocations under
# --userns=keep-id / -u <uid> can use the binary without writing to root-owned
# /usr/local/bin.
RUN GOBIN=/usr/local/bin go install golang.org/x/vuln/cmd/govulncheck@${GOVULNCHECK_VERSION}

# Install the ginkgo CLI. Build-time install ensures runtime invocations under
# --userns=keep-id / -u <uid> can use the binary without writing to root-owned
# /usr/local/bin.
RUN GOBIN=/usr/local/bin go install github.com/onsi/ginkgo/v2/ginkgo@v2.33.0

# Install Helm for rendering the E2E standalone chart.
ARG HELM_VERSION=v4.3.0
RUN GOARCH=$(uname -m | sed 's/x86_64/amd64/; s/aarch64/arm64/') && \
    cd /tmp && \
    curl -sSfLO "https://get.helm.sh/helm-${HELM_VERSION}-linux-${GOARCH}.tar.gz" && \
    curl -sSfLO "https://get.helm.sh/helm-${HELM_VERSION}-linux-${GOARCH}.tar.gz.sha256sum" && \
    sha256sum -c "helm-${HELM_VERSION}-linux-${GOARCH}.tar.gz.sha256sum" && \
    tar -xzf "helm-${HELM_VERSION}-linux-${GOARCH}.tar.gz" --strip-components=1 \
      -C /usr/local/bin "linux-${GOARCH}/helm" && \
    rm "helm-${HELM_VERSION}-linux-${GOARCH}.tar.gz" "helm-${HELM_VERSION}-linux-${GOARCH}.tar.gz.sha256sum"

# Install Weaver for telemetry registry validation and generation.
ARG WEAVER_VERSION=0.27.0
RUN apt-get update && apt-get install -y xz-utils && apt-get clean all && \
    case "$(uname -m)" in \
      x86_64) PLATFORM=x86_64-unknown-linux-gnu; CHECKSUM=7e497eb48215a5240a0bf77747cf431596c468f1039d481807b1f1e096cb5ac2 ;; \
      aarch64) PLATFORM=aarch64-unknown-linux-gnu; CHECKSUM=5bb1ebc3c5c06ee0ceac8c65243805fe819b1484cb2ca1c22d9941f53c989962 ;; \
      *) echo "Unsupported Weaver architecture: $(uname -m)" >&2; exit 1 ;; \
    esac && \
    cd /tmp && \
    curl -sSfLO "https://github.com/open-telemetry/weaver/releases/download/v${WEAVER_VERSION}/weaver-${PLATFORM}.tar.xz" && \
    echo "${CHECKSUM}  weaver-${PLATFORM}.tar.xz" | sha256sum -c - && \
    tar -xJf "weaver-${PLATFORM}.tar.xz" --strip-components=1 \
      -C /usr/local/bin "weaver-${PLATFORM}/weaver" && \
    rm "weaver-${PLATFORM}.tar.xz"

# Go caches are mounted as volumes at runtime for persistence across image rebuilds.
# Directories are created with open permissions so non-root users (docker -u) can write.
ENV GOMODCACHE=/go/pkg/mod
ENV GOCACHE=/go/cache
RUN mkdir -p /go/pkg/mod /go/cache && chmod -R 777 /go

RUN { \
      echo '#!/bin/sh'; \
      echo 'export HOME=/tmp'; \
      echo 'exec "$@"'; \
    } > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
