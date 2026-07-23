# Dockerfile demonstrating how to build a custom Endpoint Picker (EPP)
# container image with an out-of-tree custom scorer plugin compiled in.

FROM golang:1.25 as builder

WORKDIR /workspace

# Copy go.mod and go.sum
COPY go.mod go.sum ./
RUN go mod download

# Copy the entire codebase
COPY . .

# Build the custom EPP binary containing the out-of-tree scorer plugin
RUN CGO_ENABLED=0 GOOS=linux go build -a -o custom-epp ./examples/customscorer/cmd/main.go

FROM gcr.io/distroless/static:nonroot
WORKDIR /
COPY --from=builder /workspace/custom-epp .
USER 65532:65532

ENTRYPOINT ["/custom-epp"]
