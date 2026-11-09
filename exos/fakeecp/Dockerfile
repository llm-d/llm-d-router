# Builds the fake-ECP image. Build context is the repository root:
#   docker build -f exos/fakeecp/Dockerfile -t <image> .
FROM --platform=${BUILDPLATFORM} golang:1.27.3 AS go-builder
ARG TARGETOS
ARG TARGETARCH
WORKDIR /workspace
COPY go.mod go.sum ./
RUN go mod download
COPY exos/fakeecp/ exos/fakeecp/
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -ldflags="-s -w" -o /out/fakeecp ./exos/fakeecp

FROM gcr.io/distroless/static:nonroot
COPY --from=go-builder /out/fakeecp /app/fakeecp
USER 65532:65532
ENTRYPOINT ["/app/fakeecp"]
