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

	"sigs.k8s.io/controller-runtime/pkg/certwatcher"
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

// certCheckInterval throttles the cert re-read, rotations are far rarer than scrapes.
const certCheckInterval = time.Minute

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

	// refreshCert re-reads the mTLS client cert, otherwise a no-op.
	refreshCert func(context.Context)

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
	refreshCert := noopCertRefresh
	if scheme == "https" {
		rt := newReloadingTransport()
		tlsCfg, refresh, err := tlsClientConfig(tlsOpts, rt.reload)
		if err != nil {
			return nil, err
		}
		rt.tlsCfg = tlsCfg
		rt.reload()
		cl.Transport = rt
		refreshCert = refresh
	}
	return &HTTPDataSource[T]{
		typedName:      fwkplugin.TypedName{Type: pluginType, Name: pluginName},
		scheme:         scheme,
		path:           path,
		portOverride:   cfg.portOverride,
		useNodeAddress: cfg.useNodeAddress,
		client:         cl,
		parser:         parser,
		refreshCert:    refreshCert,
	}, nil
}

var (
	ErrReadCACert     = errors.New("reading CA cert")
	ErrNoValidCACerts = errors.New("no valid CA certs")
	ErrLoadClientCert = errors.New("loading client cert")
)

// caCertPool loads a PEM CA bundle for TLS verification.
func caCertPool(path string) (*x509.CertPool, error) {
	pem, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("%w %s: %w", ErrReadCACert, path, err)
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(pem) {
		return nil, fmt.Errorf("%w in %s", ErrNoValidCACerts, path)
	}
	return pool, nil
}

// noopCertRefresh is the refresher used when there is no client cert to reload.
func noopCertRefresh(context.Context) {}

// reloadingTransport swaps in a fresh transport on rotation, closing idle
// connections would leave a busy one on its old-cert handshake indefinitely.
type reloadingTransport struct {
	tlsCfg *tls.Config // set once, before the first reload that uses it
	cur    atomic.Pointer[http.Transport]
}

func newReloadingTransport() *reloadingTransport {
	t := &reloadingTransport{}
	t.cur.Store(baseTransport.Clone())
	return t
}

func (t *reloadingTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	return t.cur.Load().RoundTrip(req)
}

// reload retires the current transport, draining its connections in the background.
func (t *reloadingTransport) reload() {
	next := baseTransport.Clone()
	next.TLSClientConfig = t.tlsCfg
	t.cur.Swap(next).CloseIdleConnections()
}

// tlsClientConfig builds a tls.Config: server verification via CACertPath (or the
// system pool), plus an mTLS client certificate when ClientCertPath is set.
// The returned func reloads that certificate and calls onCertReload, the CA does not.
// onCertReload fires once during setup, before the caller has finished wiring.
func tlsClientConfig(opts TLSOptions, onCertReload func()) (*tls.Config, func(context.Context), error) {
	cfg := &tls.Config{InsecureSkipVerify: opts.SkipVerify, MinVersion: tls.VersionTLS12}
	if !opts.SkipVerify && opts.CACertPath != "" {
		pool, err := caCertPool(opts.CACertPath)
		if err != nil {
			return nil, nil, err
		}
		cfg.RootCAs = pool
	}
	if opts.ClientCertPath == "" && opts.ClientKeyPath == "" {
		return cfg, noopCertRefresh, nil
	}
	watcher, err := certwatcher.New(opts.ClientCertPath, opts.ClientKeyPath)
	if err != nil {
		return nil, nil, fmt.Errorf("%w: %w", ErrLoadClientCert, err)
	}
	// Cached so handshakes do not take certwatcher's lock, RegisterCallback seeds it.
	var current atomic.Pointer[tls.Certificate]
	watcher.RegisterCallback(func(cert tls.Certificate) {
		current.Store(&cert)
		onCertReload()
	})
	cfg.GetClientCertificate = func(*tls.CertificateRequestInfo) (*tls.Certificate, error) {
		cert := current.Load()
		if cert == nil {
			// crypto/tls dereferences the result, so a nil would panic the dial.
			return nil, fmt.Errorf("%w: no certificate loaded", ErrLoadClientCert)
		}
		return cert, nil
	}
	return cfg, throttledReload(watcher), nil
}

// throttledReload re-reads the cert at most once per certCheckInterval.
func throttledReload(watcher *certwatcher.CertWatcher) func(context.Context) {
	var lastCheck atomic.Int64
	return func(ctx context.Context) {
		last, now := lastCheck.Load(), time.Now().UnixNano()
		if now-last < int64(certCheckInterval) || !lastCheck.CompareAndSwap(last, now) {
			return
		}
		if err := watcher.ReadCertificate(); err != nil {
			log.FromContext(ctx).Error(err, "failed to re-read scrape client certificate")
		}
	}
}

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
	s.refreshCert(ctx)
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
