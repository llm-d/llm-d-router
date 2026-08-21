/*
Copyright 2025 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package tracing

import (
	"context"
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"math/big"
	"net"
	"os"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr/testr"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	coltracepb "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"
)

// collector records the spans and request metadata an exporter delivers.
type collector struct {
	coltracepb.UnimplementedTraceServiceServer

	mu    sync.Mutex
	spans int
	md    metadata.MD
}

func (c *collector) Export(ctx context.Context, req *coltracepb.ExportTraceServiceRequest) (*coltracepb.ExportTraceServiceResponse, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.md, _ = metadata.FromIncomingContext(ctx)
	for _, rs := range req.GetResourceSpans() {
		for _, ss := range rs.GetScopeSpans() {
			c.spans += len(ss.GetSpans())
		}
	}
	return &coltracepb.ExportTraceServiceResponse{}, nil
}

func (c *collector) received() (int, metadata.MD) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.spans, c.md
}

// startCollector serves the OTLP trace service on localhost. When creds is nil
// the listener is plaintext.
func startCollector(t *testing.T, creds credentials.TransportCredentials) (*collector, string) {
	t.Helper()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}

	var opts []grpc.ServerOption
	if creds != nil {
		opts = append(opts, grpc.Creds(creds))
	}
	srv := grpc.NewServer(opts...)
	c := &collector{}
	coltracepb.RegisterTraceServiceServer(srv, c)

	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	return c, lis.Addr().String()
}

// selfSignedCert returns TLS credentials for the server and the path to the PEM
// encoded certificate a client trusts to reach it.
func selfSignedCert(t *testing.T) (credentials.TransportCredentials, string) {
	t.Helper()

	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		t.Fatalf("generate key: %v", err)
	}

	tmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "otlp-test"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
		IPAddresses:           []net.IP{net.ParseIP("127.0.0.1")},
	}

	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		t.Fatalf("create certificate: %v", err)
	}

	certPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	keyDER, err := x509.MarshalECPrivateKey(key)
	if err != nil {
		t.Fatalf("marshal key: %v", err)
	}
	keyPEM := pem.EncodeToMemory(&pem.Block{Type: "EC PRIVATE KEY", Bytes: keyDER})

	pair, err := tls.X509KeyPair(certPEM, keyPEM)
	if err != nil {
		t.Fatalf("key pair: %v", err)
	}

	path := filepath.Join(t.TempDir(), "ca.pem")
	if err := os.WriteFile(path, certPEM, 0o600); err != nil {
		t.Fatalf("write certificate: %v", err)
	}

	return credentials.NewServerTLSFromCert(&pair), path
}

// exportOneSpan builds an exporter from the environment and sends a single span,
// returning the export error so transport failures are visible to the caller.
func exportOneSpan(ctx context.Context, t *testing.T) error {
	t.Helper()

	exporter, err := initTraceExporter(ctx, testr.New(t))
	if err != nil {
		t.Fatalf("initTraceExporter() error = %v", err)
	}
	t.Cleanup(func() { _ = exporter.Shutdown(context.Background()) })

	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	_, span := tp.Tracer("test").Start(ctx, "test-span")
	span.End()

	return exporter.ExportSpans(ctx, recorder.Ended())
}

// Without a certificate in the environment nothing populates gRPC credentials,
// so a pinned plaintext transport is the only reason an https endpoint would
// reach a plaintext collector. The export must fail instead.
func TestOTLPExporterDoesNotForceInsecure(t *testing.T) {
	c, addr := startCollector(t, nil)

	t.Setenv("OTEL_TRACES_EXPORTER", "otlp")
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://"+addr)
	t.Setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "2000")

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	if err := exportOneSpan(ctx, t); err == nil {
		t.Error("ExportSpans() to a plaintext collector over an https endpoint succeeded, want failure")
	}

	if spans, _ := c.received(); spans != 0 {
		t.Errorf("collector received %d spans over plaintext, want 0", spans)
	}
}

// A certificate in the environment must be used to reach a TLS collector.
func TestOTLPExporterUsesTLSFromEnvironment(t *testing.T) {
	creds, caPath := selfSignedCert(t)
	c, addr := startCollector(t, creds)

	t.Setenv("OTEL_TRACES_EXPORTER", "otlp")
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://"+addr)
	t.Setenv("OTEL_EXPORTER_OTLP_CERTIFICATE", caPath)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	if err := exportOneSpan(ctx, t); err != nil {
		t.Fatalf("ExportSpans() over TLS error = %v", err)
	}

	if spans, _ := c.received(); spans != 1 {
		t.Errorf("collector received %d spans, want 1", spans)
	}
}

// An http scheme endpoint keeps the plaintext path working.
func TestOTLPExporterStaysPlaintextForHTTPScheme(t *testing.T) {
	c, addr := startCollector(t, nil)

	t.Setenv("OTEL_TRACES_EXPORTER", "otlp")
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://"+addr)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	if err := exportOneSpan(ctx, t); err != nil {
		t.Fatalf("ExportSpans() error = %v", err)
	}

	if spans, _ := c.received(); spans != 1 {
		t.Errorf("collector received %d spans, want 1", spans)
	}
}

// OTEL_EXPORTER_OTLP_INSECURE takes precedence over the scheme of the endpoint,
// so an https endpoint marked insecure reaches a plaintext collector.
func TestOTLPExporterHonoursInsecureEnvironment(t *testing.T) {
	c, addr := startCollector(t, nil)

	t.Setenv("OTEL_TRACES_EXPORTER", "otlp")
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://"+addr)
	t.Setenv("OTEL_EXPORTER_OTLP_INSECURE", "true")

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	if err := exportOneSpan(ctx, t); err != nil {
		t.Fatalf("ExportSpans() error = %v", err)
	}

	if spans, _ := c.received(); spans != 1 {
		t.Errorf("collector received %d spans, want 1", spans)
	}
}

// Authentication headers must reach the backend.
func TestOTLPExporterSendsHeadersFromEnvironment(t *testing.T) {
	c, addr := startCollector(t, nil)

	t.Setenv("OTEL_TRACES_EXPORTER", "otlp")
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://"+addr)
	t.Setenv("OTEL_EXPORTER_OTLP_HEADERS", "authorization=Bearer test-token,x-tenant=llm-d")

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	if err := exportOneSpan(ctx, t); err != nil {
		t.Fatalf("ExportSpans() error = %v", err)
	}

	_, md := c.received()
	for key, want := range map[string]string{
		"authorization": "Bearer test-token",
		"x-tenant":      "llm-d",
	} {
		if got := md.Get(key); len(got) != 1 || got[0] != want {
			t.Errorf("metadata %q = %v, want [%q]", key, got, want)
		}
	}
}
