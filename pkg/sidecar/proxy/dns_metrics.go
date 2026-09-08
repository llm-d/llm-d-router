/*
Copyright 2026 The llm-d Authors.

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

package proxy

import (
	"context"
	"crypto/tls"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"golang.org/x/sync/errgroup"
	crmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/llm-d/llm-d-router/pkg/common"
)

// envMoRIIOMetricsAddr is a backward-compatible fallback for enabling the
// Prometheus scrape endpoint. When set to a listen address (e.g. ":9090") and
// the --metrics-port flag is unset, the sidecar serves the shared
// controller-runtime metrics registry (which carries the moriio_dns_* counters)
// at /metrics on that address. The --metrics-port flag takes precedence. Empty
// (with no flag) disables it. Kept on a separate address so it never clashes
// with the data-plane proxy port.
const envMoRIIOMetricsAddr = "MORIIO_METRICS_ADDR"

// moriioDNSSubsystem is the Prometheus subsystem prefix for the MoRI-IO
// request-path DNS re-resolution metrics. Full names are
// moriio_dns_reresolve_total, moriio_dns_ip_changed_total, and
// moriio_dns_lookup_failures_total.
const moriioDNSSubsystem = "moriio_dns"

var (
	// dnsReresolveTotal counts successful request-path DNS re-resolutions of
	// MoRI-IO peer host specs (one per actual lookup, not per request; lookups
	// coalesced by singleflight count once).
	dnsReresolveTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem: moriioDNSSubsystem,
		Name:      "reresolve_total",
		Help:      "Total successful request-path re-resolutions of MoRI-IO peer DNS names.",
	})

	// dnsIPChangedTotal counts re-resolutions where the resolved IP differed
	// from the previously cached IP (peer pod likely restarted at a new IP).
	dnsIPChangedTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem: moriioDNSSubsystem,
		Name:      "ip_changed_total",
		Help:      "Total times a MoRI-IO peer DNS name re-resolved to a different IP than the cached value.",
	})

	// dnsLookupFailuresTotal counts failed request-path DNS lookups (which then
	// fall back to the last-known-good IP or, on cold start, the raw spec).
	dnsLookupFailuresTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Subsystem: moriioDNSSubsystem,
		Name:      "lookup_failures_total",
		Help:      "Total failed MoRI-IO peer DNS lookups on the request path.",
	})

	registerDNSMetricsOnce sync.Once
)

// registerDNSMetrics registers the MoRI-IO DNS re-resolution counters on the
// controller-runtime metrics registry (the same registry the rest of the
// process exposes at /metrics). Guarded by a sync.Once so repeated resolver
// construction (including in tests) never double-registers.
func registerDNSMetrics() {
	registerDNSMetricsOnce.Do(func() {
		crmetrics.Registry.MustRegister(
			dnsReresolveTotal,
			dnsIPChangedTotal,
			dnsLookupFailuresTotal,
		)
	})
}

// metricsAddr returns the listen address for the /metrics endpoint, or "" when
// disabled. The --metrics-port flag (Config.MetricsPort) takes precedence; the
// MORIIO_METRICS_ADDR env var is a backward-compatible fallback.
func (s *Server) metricsAddr() string {
	if s.config.MetricsPort > 0 {
		return fmt.Sprintf(":%d", s.config.MetricsPort)
	}
	return strings.TrimSpace(os.Getenv(envMoRIIOMetricsAddr))
}

// maybeStartMetrics starts an opt-in Prometheus /metrics HTTP server when a
// metrics address is configured (via --metrics-port or MORIIO_METRICS_ADDR),
// registering the goroutine on grp so it shares the server lifecycle and shuts
// down with ctx. When neither is set (the default) it is a no-op and the
// counters simply go unscraped. A metrics server failure (for example an
// invalid --metrics-cert-path) is logged and never propagated to grp: the
// data-plane proxy keeps running without a /metrics endpoint rather than
// going down over an optional feature.
func (s *Server) maybeStartMetrics(ctx context.Context, grp *errgroup.Group) {
	addr := s.metricsAddr()
	if addr == "" {
		return
	}
	grp.Go(func() error {
		if err := s.serveMetrics(ctx, addr); err != nil {
			s.logger.Error(err, "metrics server failed; proxy continues without metrics")
		}
		return nil
	})
}

// serveMetrics serves the shared controller-runtime metrics registry at
// /metrics on addr until ctx is cancelled, then shuts the server down
// gracefully. Registration of the moriio_dns_* counters is ensured here so they
// are present even if no resolver has been constructed yet. The metrics server
// uses HTTP by default. When --metrics-cert-path is set, or metrics-cert-path
// is set in the sidecar YAML, it serves /metrics over HTTPS using tls.crt and
// tls.key from that directory, with no fallback to HTTP. Startup and serving
// errors are returned to maybeStartMetrics, which logs them. The shutdown
// goroutine logs shutdown errors.
func (s *Server) serveMetrics(ctx context.Context, addr string) error {
	registerDNSMetrics()

	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(crmetrics.Registry, promhttp.HandlerOpts{}))
	server := &http.Server{
		Addr:              addr,
		Handler:           mux,
		ReadHeaderTimeout: 10 * time.Second,
	}

	serveTLS := s.config.MetricsCertPath != ""
	if serveTLS {
		tlsConfig, err := s.metricsTLSConfig(ctx)
		if err != nil {
			return err
		}
		server.TLSConfig = tlsConfig
	}

	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			s.logger.Error(err, "failed to gracefully shut down metrics server")
		}
	}()

	s.logger.Info("starting MoRI-IO metrics server", "addr", addr, "tls", serveTLS)
	var err error
	if serveTLS {
		err = server.ListenAndServeTLS("", "")
	} else {
		err = server.ListenAndServe()
	}
	if err != nil && err != http.ErrServerClosed {
		return err
	}
	return nil
}

// metricsTLSConfig loads tls.crt and tls.key from Config.MetricsCertPath for
// the HTTPS metrics server. Changes to either file take effect without
// restarting the sidecar.
func (s *Server) metricsTLSConfig(ctx context.Context) (*tls.Config, error) {
	certFile := s.config.MetricsCertPath + "/tls.crt"
	keyFile := s.config.MetricsCertPath + "/tls.key"
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, fmt.Errorf("failed to load metrics TLS key pair from cert %q and key %q: %w", certFile, keyFile, err)
	}

	reloader, err := common.NewCertReloader(ctx, s.config.MetricsCertPath, &cert)
	if err != nil {
		return nil, fmt.Errorf("failed to start metrics cert reloader: %w", err)
	}

	return &tls.Config{
		MinVersion: tls.VersionTLS12,
		GetCertificate: func(*tls.ClientHelloInfo) (*tls.Certificate, error) {
			return reloader.Get(), nil
		},
	}, nil
}
