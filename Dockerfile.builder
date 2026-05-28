FROM quay.io/projectquay/golang:1.25

RUN mkdir /app
WORKDIR /app

ARG TYPOS_VERSION=v1.34.0
ARG KIND_VERSION=v0.27.0
ARG GOLANGCI_LINT_VERSION=v2.8.0
ARG KUBECTL_VERSION=v1.35.3
ARG KUSTOMIZE_VERSION=v5.6.0
ARG DOCKER_VERSION=29.3.0
ARG DOCKER_BUILDX_VERSION=v0.32.1
ARG ENVTEST_VERSION=release-0.19
ARG ENVTEST_K8S_VERSION=1.31.0
ARG GOVULNCHECK_VERSION=v1.3.0

RUN dnf install -y podman gcc-toolset-12 && dnf clean all

# The base image ships GCC 8 (RHEL 8 default), which lacks the ARM64 LSE
# atomic emulation helpers (__aarch64_ldadd8_sync etc.) required by Go's
# race detector. gcc-toolset-12 provides GCC 12 which includes them.
# TODO: remove gcc-toolset-12 once this base image is updated to RHEL 9
#       (GCC 11+ ships natively there and includes the helpers).
ENV PATH=/opt/rh/gcc-toolset-12/root/usr/bin:$PATH

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
