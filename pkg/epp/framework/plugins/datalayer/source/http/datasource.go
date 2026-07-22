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

package http

import (
	"context"
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"reflect"
	"runtime/debug"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

var ErrExtractorTypeMismatch = errors.New("extractor type mismatch")

// defaultStepTimeout bounds each Poll and each Extract independently so a slow
// extractor cannot starve sibling extractors of their tick budget.
const defaultStepTimeout = time.Second

// HTTPDataSource is a typed polling dispatcher. T is the data type the source
// produces; bound extractors must implement Extractor[PollInput[T]].
type HTTPDataSource[T any] struct {
	typedName fwkplugin.TypedName
	scheme    string
	path      string
	// portOverride, when non-zero, replaces the port in the endpoint's
	// MetricsHost with this value. This allows a source to target a
	// different port on the same pod (e.g. DCGM Exporter on :9400)
	// without changing the endpoint metadata set by the discovery layer.
	portOverride int
	// useNodeAddress, when true, scrapes NodeAddress:portOverride instead
	// of the pod IP. Used for node-level exporters (e.g. DCGM DaemonSet).
	useNodeAddress bool

	client Client
	// parser converts the response body to T. MUST NOT return (zero, nil) for nilable T;
	// the dispatcher does not validate.
	parser func(io.Reader) (T, error)

	mu   sync.RWMutex
	exts []fwkdl.PollingExtractor[T]
}

// TLSOptions configures the https transport. The zero value verifies the target
// against the system CA pool with no client certificate.
type TLSOptions struct {
	// SkipVerify disables verification of the target's server certificate.
	SkipVerify bool
	// CACertPath is a PEM CA bundle used to verify the target instead of the
	// system pool. Ignored when SkipVerify is set.
	CACertPath string
	// ClientCertPath and ClientKeyPath present a client certificate for mTLS.
	// Both must be set together.
	ClientCertPath string
	ClientKeyPath  string
}

// Option configures optional behaviour on an HTTPDataSource.
type Option func(*options)

type options struct {
	portOverride   int
	useNodeAddress bool
}

// WithPortOverride makes the source scrape podIP:port instead of the
// endpoint's MetricsHost. Use this when a sidecar (e.g. DCGM Exporter)
// listens on a different port than the inference server.
func WithPortOverride(port int) Option {
	return func(o *options) { o.portOverride = port }
}

// WithUseNodeAddress makes the source scrape nodeIP:portOverride instead
// of podIP:portOverride. Requires a non-zero portOverride and a non-empty
// NodeAddress on the endpoint metadata.
func WithUseNodeAddress() Option {
	return func(o *options) { o.useNodeAddress = true }
}

// NewHTTPDataSource constructs a typed polling dispatcher. For https, tlsOpts configures
// server verification (CACertPath) and optional mTLS (ClientCertPath/ClientKeyPath).
func NewHTTPDataSource[T any](scheme, path string, tlsOpts TLSOptions,
	pluginType, pluginName string, parser func(io.Reader) (T, error),
	opts ...Option) (*HTTPDataSource[T], error) {
	if scheme != "http" && scheme != "https" {
		return nil, fmt.Errorf("unsupported scheme: %s", scheme)
	}

	var cfg options
	for _, o := range opts {
		o(&cfg)
	}
	if cfg.useNodeAddress && cfg.portOverride == 0 {
		return nil, errors.New("WithUseNodeAddress requires a non-zero WithPortOverride")
	}

	cl := &client{
		Client: http.Client{
			Timeout:   timeout,
			Transport: baseTransport,
		},
	}
	if scheme == "https" {
		rt, err := newTLSReloader(tlsOpts)
		if err != nil {
			return nil, err
		}
		cl.Transport = rt
	}
	return &HTTPDataSource[T]{
		typedName:      fwkplugin.TypedName{Type: pluginType, Name: pluginName},
		scheme:         scheme,
		path:           path,
		portOverride:   cfg.portOverride,
		useNodeAddress: cfg.useNodeAddress,
		client:         cl,
		parser:         parser,
	}, nil
}

// tlsReloadInterval bounds how often the scrape client re-reads its TLS files.
// Rotation is rare, so a coarse interval keeps the check off the hot scrape path.
const tlsReloadInterval = 10 * time.Second

// tlsRetryInterval is the shorter re-check after a failed read, so a transient error
// (a partial write mid-rotation) recovers without waiting a full interval or coupling
// retry cadence to scrape volume.
const tlsRetryInterval = time.Second

// tlsReloader is the https scrape transport. It presents the client certificate and
// verifies the server against a CA bundle, reloading both from disk when the files
// change: the client cert is served through GetClientCertificate, a changed CA
// rebuilds the transport with a fresh RootCAs pool. Verification stays with the
// standard library, so a rotated bundle applies without a restart.
type tlsReloader struct {
	caPath, certPath, keyPath string
	skipVerify                bool

	base      time.Time    // monotonic anchor for nextCheck
	nextCheck atomic.Int64 // monotonic nanos since base gating the next file read

	reloadMu                  sync.Mutex // serializes a reload
	caHash, certHash, keyHash [sha256.Size]byte

	cert    atomic.Pointer[tls.Certificate] // current client cert, nil when none configured
	current atomic.Pointer[http.Transport]  // transport in use, swapped on CA change
}

// newTLSReloader loads the initial CA and client certificate, failing if either is
// configured but unreadable or invalid.
func newTLSReloader(opts TLSOptions) (*tlsReloader, error) {
	t := &tlsReloader{
		caPath:     opts.CACertPath,
		certPath:   opts.ClientCertPath,
		keyPath:    opts.ClientKeyPath,
		skipVerify: opts.SkipVerify,
		base:       time.Now(),
	}
	var pool *x509.CertPool
	if t.caPath != "" && !t.skipVerify {
		data, err := os.ReadFile(t.caPath)
		if err != nil {
			return nil, fmt.Errorf("%w %s: %w", ErrReadCACert, t.caPath, err)
		}
		pool = x509.NewCertPool()
		if !pool.AppendCertsFromPEM(data) {
			return nil, fmt.Errorf("%w in %s", ErrNoValidCACerts, t.caPath)
		}
		t.caHash = sha256.Sum256(data)
	}
	if t.certPath != "" || t.keyPath != "" {
		certData, keyData, err := t.readCert()
		if err != nil {
			return nil, err
		}
		if err := t.storeCert(certData, keyData); err != nil {
			return nil, err
		}
	}
	t.current.Store(t.buildTransport(pool))
	t.nextCheck.Store(t.since() + tlsReloadInterval.Nanoseconds())
	return t, nil
}

func (t *tlsReloader) readCert() (cert, key []byte, err error) {
	if cert, err = os.ReadFile(t.certPath); err != nil {
		return nil, nil, fmt.Errorf("%w: %w", ErrLoadClientCert, err)
	}
	if key, err = os.ReadFile(t.keyPath); err != nil {
		return nil, nil, fmt.Errorf("%w: %w", ErrLoadClientCert, err)
	}
	return cert, key, nil
}

// storeCert parses the keypair into the served pointer and records the file hashes.
func (t *tlsReloader) storeCert(certData, keyData []byte) error {
	pair, err := tls.X509KeyPair(certData, keyData)
	if err != nil {
		return fmt.Errorf("%w: %w", ErrLoadClientCert, err)
	}
	t.cert.Store(&pair)
	t.certHash, t.keyHash = sha256.Sum256(certData), sha256.Sum256(keyData)
	return nil
}

// buildTransport clones the base transport with a tls.Config that verifies the server
// against pool (nil means the system pool) and presents the current client cert.
func (t *tlsReloader) buildTransport(pool *x509.CertPool) *http.Transport {
	cfg := &tls.Config{InsecureSkipVerify: t.skipVerify, RootCAs: pool}
	if t.certPath != "" {
		cfg.GetClientCertificate = func(*tls.CertificateRequestInfo) (*tls.Certificate, error) {
			return t.cert.Load(), nil
		}
	}
	tr := baseTransport.Clone()
	tr.TLSClientConfig = cfg
	return tr
}

func (t *tlsReloader) RoundTrip(req *http.Request) (*http.Response, error) {
	t.maybeReload(req.Context())
	return t.current.Load().RoundTrip(req)
}

func (t *tlsReloader) CloseIdleConnections() { t.current.Load().CloseIdleConnections() }

// since returns monotonic nanoseconds elapsed since the reloader was created, so
// the throttle is unaffected by wall-clock steps (NTP correction, live-migration).
func (t *tlsReloader) since() int64 { return int64(time.Since(t.base)) }

// maybeReload re-reads the TLS files at most once per interval and applies what
// changed. Reload is scrape-driven: an unscraped endpoint never reloads, which is
// fine since it presents the cert to no one. On a read or parse error it keeps the
// last-good material (a partial write never drops verification) and retries after a
// short backoff.
func (t *tlsReloader) maybeReload(ctx context.Context) {
	elapsed := t.since()
	next := t.nextCheck.Load()
	if elapsed < next {
		return
	}
	if !t.nextCheck.CompareAndSwap(next, elapsed+tlsReloadInterval.Nanoseconds()) {
		return
	}
	t.reloadMu.Lock()
	defer t.reloadMu.Unlock()
	ok := true
	if t.certPath != "" {
		ok = t.reloadCert(ctx) && ok
	}
	if t.caPath != "" && !t.skipVerify {
		ok = t.reloadCA(ctx) && ok
	}
	if !ok {
		t.nextCheck.Store(t.since() + tlsRetryInterval.Nanoseconds()) // read failed, retry soon
	}
}

// reloadCert reports whether the files were read successfully (a no-op re-read
// counts as success). A false return drives a retry on the next scrape.
func (t *tlsReloader) reloadCert(ctx context.Context) bool {
	logger := log.FromContext(ctx)
	certData, keyData, err := t.readCert()
	if err != nil {
		metrics.LlmdDataLayerTLSReloadErrorsTotal.WithLabelValues("cert").Inc()
		logger.Error(err, "client cert reload read failed, keeping current cert")
		return false
	}
	if sha256.Sum256(certData) == t.certHash && sha256.Sum256(keyData) == t.keyHash {
		return true
	}
	if err := t.storeCert(certData, keyData); err != nil {
		metrics.LlmdDataLayerTLSReloadErrorsTotal.WithLabelValues("cert").Inc()
		logger.Error(err, "client cert reload parse failed, keeping current cert")
		return false
	}
	t.current.Load().CloseIdleConnections() // re-handshake so keep-alives present the new cert
	logger.Info("client certificate reloaded")
	return true
}

// reloadCA reports whether the bundle was read successfully. A false return drives
// a retry on the next scrape.
func (t *tlsReloader) reloadCA(ctx context.Context) bool {
	logger := log.FromContext(ctx)
	data, err := os.ReadFile(t.caPath)
	if err != nil {
		metrics.LlmdDataLayerTLSReloadErrorsTotal.WithLabelValues("ca").Inc()
		logger.Error(err, "CA reload read failed, keeping current bundle")
		return false
	}
	h := sha256.Sum256(data)
	if h == t.caHash {
		return true
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(data) {
		metrics.LlmdDataLayerTLSReloadErrorsTotal.WithLabelValues("ca").Inc()
		logger.Error(ErrNoValidCACerts, "CA reload parse failed, keeping current bundle")
		return false
	}
	old := t.current.Swap(t.buildTransport(pool))
	t.caHash = h
	old.CloseIdleConnections()
	logger.Info("CA bundle reloaded")
	return true
}

var (
	ErrReadCACert     = errors.New("reading CA cert")
	ErrNoValidCACerts = errors.New("no valid CA certs")
	ErrLoadClientCert = errors.New("loading client cert")
)

func (s *HTTPDataSource[T]) TypedName() fwkplugin.TypedName { return s.typedName }

// Poll fetches and parses one tick. Exposed for tests; runtime uses Dispatch.
func (s *HTTPDataSource[T]) Poll(ctx context.Context, ep fwkdl.Endpoint) (T, error) {
	target := s.getEndpoint(ep.GetMetadata())
	raw, err := s.client.Get(ctx, target, ep.GetMetadata(), func(r io.Reader) (any, error) {
		return s.parser(r)
	})
	if err != nil {
		var zero T
		return zero, err
	}
	// Defensive: unreachable with the current Client (parser passthrough); remove with Client[T] refactor.
	typed, ok := raw.(T)
	if !ok {
		var zero T
		return zero, fmt.Errorf("HTTPDataSource %s: parser returned %T, expected %T", s.typedName, raw, zero)
	}
	return typed, nil
}

// Dispatch polls the endpoint and fans the result out to every bound
// extractor. Each step (Poll and each Extract) runs under its own
// defaultStepTimeout so one slow extractor does not starve siblings.
//
// Return contract: a non-nil return indicates a poll-level failure (the
// dispatcher could not produce data). Per-extractor failures are recorded
// in DataLayerExtractErrorsTotal and do NOT surface as a returned error.
// This keeps the collector's poll/extract counters cleanly separated.
func (s *HTTPDataSource[T]) Dispatch(ctx context.Context, ep fwkdl.Endpoint) error {
	pollCtx, cancelPoll := context.WithTimeout(ctx, defaultStepTimeout)
	data, err := s.Poll(pollCtx, ep)
	cancelPoll()
	if err != nil {
		return err
	}
	in := fwkdl.PollInput[T]{Payload: data, Endpoint: ep}
	s.mu.RLock()
	exts := slices.Clone(s.exts)
	s.mu.RUnlock()
	for _, ext := range exts {
		if ctx.Err() != nil {
			return nil
		}
		extCtx, cancelExt := context.WithTimeout(ctx, defaultStepTimeout)
		s.runExtractor(extCtx, ext, in)
		cancelExt()
	}
	return nil
}

// runExtractor invokes ext under panic recovery; both failures and panics increment DataLayerExtractErrorsTotal.
func (s *HTTPDataSource[T]) runExtractor(ctx context.Context, ext fwkdl.PollingExtractor[T], in fwkdl.PollInput[T]) {
	logger := log.FromContext(ctx)
	srcType := s.typedName.Type
	extType := ext.TypedName().Type
	defer func() {
		if r := recover(); r != nil {
			metrics.RecordDataLayerExtractError(srcType, extType)
			logger.Error(fmt.Errorf("%v", r), "extractor panicked",
				"source", s.typedName, "extractor", ext.TypedName(), "stack", string(debug.Stack()))
		}
	}()
	if err := ext.Extract(ctx, in); err != nil {
		metrics.RecordDataLayerExtractError(srcType, extType)
		logger.V(logging.DEBUG).Info("extract failed", "source", s.typedName, "extractor", ext.TypedName(), "err", err)
	}
}

// AppendExtractor binds ext as a typed PollingExtractor[T]. Duplicate-Type detection
// is the caller's responsibility (see runtime.Configure); this is a pure append.
func (s *HTTPDataSource[T]) AppendExtractor(ext fwkplugin.Plugin) error {
	typed, ok := ext.(fwkdl.PollingExtractor[T])
	if !ok {
		return fmt.Errorf("%w: extractor %s: expected %s, got %T",
			ErrExtractorTypeMismatch, ext.TypedName(), reflect.TypeFor[fwkdl.PollingExtractor[T]](), ext)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.exts = append(s.exts, typed)
	return nil
}

func (s *HTTPDataSource[T]) getEndpoint(ep Addressable) *url.URL {
	host := ep.GetMetricsHost()
	if s.portOverride > 0 {
		ip := ep.GetIPAddress()
		if s.useNodeAddress {
			if nodeIP := ep.GetNodeAddress(); nodeIP != "" {
				ip = nodeIP
			}
		}
		host = net.JoinHostPort(ip, strconv.Itoa(s.portOverride))
	}
	return &url.URL{Scheme: s.scheme, Host: host, Path: s.path}
}
