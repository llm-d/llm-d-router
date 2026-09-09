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
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr/funcr"
	"github.com/go-logr/logr/testr"
	"go.opentelemetry.io/otel"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	coltracepb "go.opentelemetry.io/proto/otlp/collector/trace/v1"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/metadata"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
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

	return serveCollector(t, creds, lis), lis.Addr().String()
}

// serveCollector runs the OTLP trace service on lis. When creds is nil the
// listener is plaintext.
func serveCollector(t *testing.T, creds credentials.TransportCredentials, lis net.Listener) *collector {
	t.Helper()

	var opts []grpc.ServerOption
	if creds != nil {
		opts = append(opts, grpc.Creds(creds))
	}
	srv := grpc.NewServer(opts...)
	c := &collector{}
	coltracepb.RegisterTraceServiceServer(srv, c)

	go func() { _ = srv.Serve(lis) }()
	t.Cleanup(srv.Stop)

	return c
}

// selfSignedCert returns TLS credentials for the server and the path to the PEM
// encoded certificate a client trusts to reach it.
func selfSignedCert(t *testing.T) (credentials.TransportCredentials, string) {
	t.Helper()

	pair, path := selfSignedKeyPair(t)
	return credentials.NewServerTLSFromCert(pair), path
}

// selfSignedKeyPair returns the certificate itself, for servers that are not gRPC,
// alongside the path of the PEM a client trusts it by.
func selfSignedKeyPair(t *testing.T) (*tls.Certificate, string) {
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

	return &pair, path
}

// exportOneSpan builds an exporter from the environment and sends a single span,
// returning the export error so transport failures are visible to the caller.
func exportOneSpan(ctx context.Context, t *testing.T) error {
	t.Helper()

	exporterType, err := traceExporterType()
	if err != nil {
		t.Fatalf("traceExporterType() error = %v", err)
	}
	exporter, err := newTraceExporter(ctx, &errorHandler{logger: testr.New(t)}, exporterType)
	if err != nil {
		t.Fatalf("newTraceExporter() error = %v", err)
	}
	t.Cleanup(func() { _ = exporter.Shutdown(context.Background()) })

	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	_, span := tp.Tracer("test").Start(ctx, "test-span")
	span.End()

	return exporter.ExportSpans(ctx, recorder.Ended())
}

// With no transport variables set the exporter must reach a plaintext collector
// on the loopback default. The SDK's own default endpoint negotiates TLS, so the
// insecure transport has to be pinned for that case.
func TestOTLPExporterDefaultsToPlaintextLoopback(t *testing.T) {
	clearEnv(t, otlpTransportEnv...)

	lis, err := net.Listen("tcp", "localhost:4317")
	if err != nil {
		t.Skipf("cannot bind the default OTLP port: %v", err)
	}
	c := serveCollector(t, nil, lis)

	t.Setenv("OTEL_TRACES_EXPORTER", "otlp")

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	if err := exportOneSpan(ctx, t); err != nil {
		t.Fatalf("ExportSpans() to the loopback default error = %v", err)
	}

	if spans, _ := c.received(); spans != 1 {
		t.Errorf("collector received %d spans, want 1", spans)
	}
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

// InitTracing with nothing configured must reach a collector, not pretty print to
// stdout. This is the whole point of the default: the loopback fallback and the
// otlp default have to line up so an unconfigured process exports rather than
// flooding its own log stream.
func TestInitTracingDefaultsToOTLPCollector(t *testing.T) {
	clearEnv(t, otlpTransportEnv...)
	clearEnv(t, "OTEL_TRACES_EXPORTER", "OTEL_TRACES_SAMPLER", "OTEL_TRACES_SAMPLER_ARG")
	t.Setenv("OTEL_TRACES_SAMPLER", "always_on")

	lis, err := net.Listen("tcp", "localhost:4317")
	if err != nil {
		t.Skipf("cannot bind the default OTLP port: %v", err)
	}
	c := serveCollector(t, nil, lis)

	origTP, origHandler, origProp := otel.GetTracerProvider(), otel.GetErrorHandler(), otel.GetTextMapPropagator()
	t.Cleanup(func() {
		otel.SetTracerProvider(origTP)
		otel.SetErrorHandler(origHandler)
		otel.SetTextMapPropagator(origProp)
	})

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	shutdown, err := InitTracing(ctx, testr.New(t), testServiceName)
	if err != nil {
		t.Fatalf("InitTracing() error = %v", err)
	}

	_, span := Tracer().Start(ctx, "default-exporter-span")
	span.End()

	// Shutdown flushes the batcher, so the collector has the span once it returns.
	if err := shutdown(ctx); err != nil {
		t.Fatalf("shutdown() error = %v", err)
	}

	if spans, _ := c.received(); spans != 1 {
		t.Errorf("collector received %d spans, want 1 exported by the default exporter", spans)
	}
}

// httpCollector records what an OTLP/HTTP exporter delivers to /v1/traces.
type httpCollector struct {
	mu       sync.Mutex
	requests int
	header   http.Header
}

func (c *httpCollector) record(r *http.Request) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.requests++
	c.header = r.Header.Clone()
}

func (c *httpCollector) received() (int, http.Header) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.requests, c.header
}

// serveHTTPCollector answers OTLP/HTTP export requests on lis. An empty protobuf
// response body is a valid ExportTraceServiceResponse, so the exporter treats the
// export as successful.
func serveHTTPCollector(t *testing.T, tlsCert *tls.Certificate, lis net.Listener) *httpCollector {
	t.Helper()

	c := &httpCollector{}
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/traces", func(w http.ResponseWriter, r *http.Request) {
		c.record(r)
		w.Header().Set("Content-Type", "application/x-protobuf")
		w.WriteHeader(http.StatusOK)
	})

	srv := &http.Server{Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	if tlsCert != nil {
		srv.TLSConfig = &tls.Config{Certificates: []tls.Certificate{*tlsCert}, MinVersion: tls.VersionTLS12}
		go func() { _ = srv.ServeTLS(lis, "", "") }()
	} else {
		go func() { _ = srv.Serve(lis) }()
	}
	t.Cleanup(func() { _ = srv.Close() })

	return c
}

// startHTTPCollector serves OTLP/HTTP on an arbitrary loopback port.
func startHTTPCollector(t *testing.T, tlsCert *tls.Certificate) (*httpCollector, string) {
	t.Helper()

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}

	return serveHTTPCollector(t, tlsCert, lis), lis.Addr().String()
}

func TestOTLPProtocol(t *testing.T) {
	tests := []struct {
		name    string
		env     map[string]string
		want    string
		wantErr bool
	}{
		{name: "unset defaults to grpc", want: protocolGRPC},
		{
			name: "grpc",
			env:  map[string]string{"OTEL_EXPORTER_OTLP_PROTOCOL": "grpc"},
			want: protocolGRPC,
		},
		{
			name: "http/protobuf",
			env:  map[string]string{"OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf"},
			want: protocolHTTP,
		},
		{
			name: "the per-signal variable wins over the generic one",
			env: map[string]string{
				"OTEL_EXPORTER_OTLP_PROTOCOL":        "grpc",
				"OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "http/protobuf",
			},
			want: protocolHTTP,
		},
		{
			name: "the generic variable applies when the per-signal one is unset",
			env:  map[string]string{"OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf"},
			want: protocolHTTP,
		},
		{
			name:    "http/json is reported as unsupported",
			env:     map[string]string{"OTEL_EXPORTER_OTLP_PROTOCOL": "http/json"},
			want:    protocolGRPC,
			wantErr: true,
		},
		{
			name: "a supported value matches case-insensitively",
			env:  map[string]string{"OTEL_EXPORTER_OTLP_PROTOCOL": "HTTP/PROTOBUF"},
			want: protocolHTTP,
		},
		{
			name: "grpc matches case-insensitively",
			env:  map[string]string{"OTEL_EXPORTER_OTLP_PROTOCOL": "GRPC"},
			want: protocolGRPC,
		},
		{
			name:    "an unrecognised protocol is reported",
			env:     map[string]string{"OTEL_EXPORTER_OTLP_PROTOCOL": "thrift"},
			want:    protocolGRPC,
			wantErr: true,
		},
		{
			name: "an empty value counts as unset",
			env:  map[string]string{"OTEL_EXPORTER_OTLP_PROTOCOL": ""},
			want: protocolGRPC,
		},
		{
			name: "a blank value counts as unset",
			env:  map[string]string{"OTEL_EXPORTER_OTLP_PROTOCOL": "   "},
			want: protocolGRPC,
		},
		{
			name: "surrounding whitespace does not reject a supported value",
			env:  map[string]string{"OTEL_EXPORTER_OTLP_PROTOCOL": "  http/protobuf  "},
			want: protocolHTTP,
		},
		{
			name: "an empty per-signal value falls through to the generic one",
			env: map[string]string{
				"OTEL_EXPORTER_OTLP_PROTOCOL":        "http/protobuf",
				"OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "",
			},
			want: protocolHTTP,
		},
		{
			name: "a rejected per-signal value is ignored so the generic one applies",
			env: map[string]string{
				"OTEL_EXPORTER_OTLP_PROTOCOL":        "http/protobuf",
				"OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "thrift",
			},
			want:    protocolHTTP,
			wantErr: true,
		},
		{
			name: "gRPC applies when no variable names a supported protocol",
			env: map[string]string{
				"OTEL_EXPORTER_OTLP_PROTOCOL":        "http/json",
				"OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": "thrift",
			},
			want:    protocolGRPC,
			wantErr: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			clearEnv(t, otlpProtocolEnv...)
			for key, value := range tc.env {
				t.Setenv(key, value)
			}

			got, err := otlpProtocol()
			if (err != nil) != tc.wantErr {
				t.Fatalf("otlpProtocol() error = %v, wantErr %v", err, tc.wantErr)
			}
			if got != tc.want {
				t.Errorf("otlpProtocol() = %q, want %q", got, tc.want)
			}
		})
	}
}

// http/protobuf must actually export over HTTP, not silently keep using gRPC.
func TestOTLPHTTPExporterReachesCollector(t *testing.T) {
	clearEnv(t, otlpTransportEnv...)
	clearEnv(t, otlpProtocolEnv...)

	c, addr := startHTTPCollector(t, nil)

	t.Setenv("OTEL_TRACES_EXPORTER", "otlp")
	t.Setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://"+addr)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	if err := exportOneSpan(ctx, t); err != nil {
		t.Fatalf("ExportSpans() over OTLP/HTTP error = %v", err)
	}

	if requests, _ := c.received(); requests != 1 {
		t.Errorf("collector received %d requests, want 1", requests)
	}
}

// The loopback fallback has to follow the protocol: the HTTP transport is served on
// a different port than gRPC, so sharing one default would reach nothing.
func TestOTLPHTTPExporterDefaultsToPlaintextLoopback(t *testing.T) {
	clearEnv(t, otlpTransportEnv...)
	clearEnv(t, otlpProtocolEnv...)

	lis, err := net.Listen("tcp", localHTTPEndpoint)
	if err != nil {
		t.Skipf("cannot bind the default OTLP/HTTP port: %v", err)
	}
	c := serveHTTPCollector(t, nil, lis)

	t.Setenv("OTEL_TRACES_EXPORTER", "otlp")
	t.Setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	if err := exportOneSpan(ctx, t); err != nil {
		t.Fatalf("ExportSpans() to the loopback default error = %v", err)
	}

	if requests, _ := c.received(); requests != 1 {
		t.Errorf("collector received %d requests, want 1", requests)
	}
}

// The authentication behaviour established for gRPC must hold on the HTTP path.
func TestOTLPHTTPExporterSendsHeadersFromEnvironment(t *testing.T) {
	clearEnv(t, otlpTransportEnv...)
	clearEnv(t, otlpProtocolEnv...)

	c, addr := startHTTPCollector(t, nil)

	t.Setenv("OTEL_TRACES_EXPORTER", "otlp")
	t.Setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://"+addr)
	t.Setenv("OTEL_EXPORTER_OTLP_HEADERS", "authorization=Bearer token123")

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	if err := exportOneSpan(ctx, t); err != nil {
		t.Fatalf("ExportSpans() error = %v", err)
	}

	_, header := c.received()
	if got := header.Get("Authorization"); got != "Bearer token123" {
		t.Errorf("Authorization header = %q, want %q", got, "Bearer token123")
	}
}

// An https endpoint must reach a TLS collector on the HTTP path, the same way it does
// over gRPC.
func TestOTLPHTTPExporterUsesTLSFromEnvironment(t *testing.T) {
	clearEnv(t, otlpTransportEnv...)
	clearEnv(t, otlpProtocolEnv...)

	cert, certPath := selfSignedKeyPair(t)

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	c := serveHTTPCollector(t, cert, lis)

	t.Setenv("OTEL_TRACES_EXPORTER", "otlp")
	t.Setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")
	t.Setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "https://"+lis.Addr().String())
	t.Setenv("OTEL_EXPORTER_OTLP_CERTIFICATE", certPath)

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()

	if err := exportOneSpan(ctx, t); err != nil {
		t.Fatalf("ExportSpans() over TLS error = %v", err)
	}

	if requests, _ := c.received(); requests != 1 {
		t.Errorf("collector received %d requests, want 1", requests)
	}
}

// A rejected protocol reaches the operator only through the error handler: the exporter
// is still built, so nothing downstream reports the value that was ignored.
func TestUnsupportedProtocolReachesErrorHandler(t *testing.T) {
	clearEnv(t, otlpTransportEnv...)
	clearEnv(t, otlpProtocolEnv...)

	t.Setenv("OTEL_EXPORTER_OTLP_PROTOCOL", "thrift")

	var handled []string
	h := &errorHandler{logger: funcr.New(func(_, args string) {
		handled = append(handled, args)
	}, funcr.Options{Verbosity: logging.DEBUG})}

	exporter, err := newTraceExporter(context.Background(), h, exporterTypeOTLP)
	if err != nil {
		t.Fatalf("newTraceExporter() error = %v", err)
	}
	t.Cleanup(func() { _ = exporter.Shutdown(context.Background()) })

	for _, line := range handled {
		if strings.Contains(line, "thrift") {
			return
		}
	}
	t.Errorf("error handler received %q, want a report naming the rejected protocol", handled)
}
