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

package main

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/spf13/pflag"
	"golang.org/x/sync/errgroup"
	ctrl "sigs.k8s.io/controller-runtime"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/llm-d/llm-d-router/pkg/common"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/version"

	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline/builder"
	"github.com/llm-d/llm-d-router/pkg/coordinator/server"
)

// metricsShutdownTimeout bounds the Prometheus /metrics server's shutdown.
// Scrapes are short; a small budget is enough and keeps process exit prompt.
const metricsShutdownTimeout = 5 * time.Second

var errMetricsTLS = errors.New("metrics TLS")

func main() {
	configPath := pflag.String("config", "config/coordinator/coordinator.yaml", "path to configuration file")
	metricsPort := pflag.Int("metrics-port", 0, "port for the Prometheus /metrics endpoint. Non-positive disables the endpoint. Overrides server.metrics_port (default 9090).")
	metricsCertPath := pflag.String("metrics-cert-path", "", "directory with tls.crt and tls.key for the metrics endpoint. Empty serves metrics over HTTP. Overrides server.metrics_cert_path.")

	logOpts := logutil.NewOptions()
	logOpts.AddFlags(pflag.CommandLine)

	pflag.Parse()

	logutil.InitSetupLogging("llm-d-coordinator")
	log := ctrl.Log.WithName("coordinator")

	log.Info("coordinator build", "commit-sha", version.CommitSHA, "build-ref", version.BuildRef)

	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Error(err, "failed to load config")
		os.Exit(1)
	}

	// CLI -v wins over config log_level.
	if vFlag := pflag.CommandLine.Lookup("v"); vFlag != nil && !vFlag.Changed {
		logOpts.LogVerbosity = cfg.LogLevel
	}
	// CLI --metrics-port wins over config server.metrics_port.
	if f := pflag.CommandLine.Lookup("metrics-port"); f != nil && f.Changed {
		cfg.Server.MetricsPort = *metricsPort
	}
	// CLI --metrics-cert-path wins over server.metrics_cert_path.
	if f := pflag.CommandLine.Lookup("metrics-cert-path"); f != nil && f.Changed {
		cfg.Server.MetricsCertPath = *metricsCertPath
	}
	if err := logOpts.Validate(); err != nil {
		log.Error(err, "invalid logging options")
		os.Exit(1)
	}
	if err := logOpts.Complete(); err != nil {
		log.Error(err, "failed to complete logging options")
		os.Exit(1)
	}
	logutil.InitLogging(&logOpts.ZapOptions)
	log.Info("log level set", "level", logOpts.LogVerbosity)
	log.Info("pipeline connectors",
		"kv_connector", cfg.Pipeline.KVConnector,
		"ec_connector", cfg.Pipeline.ECConnector)
	// Log presence only: proxy URLs can carry basic-auth credentials
	// (http://user:pass@host) and must not reach startup logs. NO_PROXY is a
	// plain host list, so it is safe to log verbatim.
	log.Info("proxy environment",
		"http_proxy_set", os.Getenv("HTTP_PROXY") != "",
		"https_proxy_set", os.Getenv("HTTPS_PROXY") != "",
		"NO_PROXY", os.Getenv("NO_PROXY"))

	if err := coordmetrics.Register(ctrlmetrics.Registry); err != nil {
		log.Error(err, "failed to register coordinator metrics")
		os.Exit(1)
	}

	gwClient := gateway.New(cfg.Gateway)

	steps, err := builder.Build(cfg, gwClient)
	if err != nil {
		log.Error(err, "failed to build pipeline")
		os.Exit(1)
	}

	p := pipeline.New(steps)
	srv, err := server.New(cfg.Server, p, gwClient)
	if err != nil {
		log.Error(err, "failed to create server")
		os.Exit(1)
	}

	log.Info("starting coordinator",
		"addr", cfg.Server.ListenAddr,
		"metrics_port", cfg.Server.MetricsPort,
		"metrics_tls", cfg.Server.MetricsCertPath != "")
	if cfg.Server.MetricsPort <= 0 {
		log.Info("metrics endpoint disabled", "reason", "server.metrics_port <= 0")
	}
	log.Info("graceful shutdown enabled", "timeout", cfg.Server.ShutdownTimeout)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	err = run(ctx, srv, cfg.Server)
	stop()
	if err != nil {
		log.Error(err, "server error")
		os.Exit(1)
	}
}

// run starts the coordinator server and, when cfg.MetricsPort > 0, the
// Prometheus /metrics server alongside it. It blocks until ctx is cancelled
// or either server exits. On any exit condition both servers are drained
// before run returns: the coordinator server bounded by cfg.ShutdownTimeout,
// the metrics server by metricsShutdownTimeout. A non-positive MetricsPort
// disables the metrics endpoint entirely.
func run(ctx context.Context, srv *server.Server, cfg config.ServerConfig) error {
	g, gctx := errgroup.WithContext(ctx)

	g.Go(func() error {
		errCh := make(chan error, 1)
		go func() { errCh <- srv.ListenAndServe() }()
		select {
		case err := <-errCh:
			// ListenAndServe returned before shutdown was requested; always a failure.
			return fmt.Errorf("coordinator server: %w", err)
		case <-gctx.Done():
			shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
			defer cancel()
			if err := srv.Shutdown(shutdownCtx); err != nil {
				return fmt.Errorf("coordinator server shutdown: %w", err)
			}
			if err := <-errCh; err != nil && !errors.Is(err, http.ErrServerClosed) {
				return fmt.Errorf("coordinator server: %w", err)
			}
			return nil
		}
	})

	if cfg.MetricsPort > 0 {
		g.Go(func() error {
			return serveMetrics(gctx, cfg.MetricsPort, cfg.MetricsCertPath)
		})
	}

	return g.Wait()
}

// serveMetrics stands up a Prometheus /metrics server on port and blocks
// until ctx is cancelled or the underlying ListenAndServe returns
// unexpectedly. On ctx cancellation the server is drained via Shutdown
// bounded by metricsShutdownTimeout. Uses the shared controller-runtime
// registry so every package that registers against it (this coordinator's
// metrics, controller-runtime's process collectors) is exposed on the same
// endpoint. A non-empty certPath enables HTTPS with tls.crt and tls.key.
func serveMetrics(ctx context.Context, port int, certPath string) error {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(ctrlmetrics.Registry, promhttp.HandlerOpts{EnableOpenMetrics: true}))
	srv := &http.Server{
		Addr:              fmt.Sprintf(":%d", port),
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	serveTLS := certPath != ""
	if serveTLS {
		tlsConfig, err := metricsTLSConfig(ctx, certPath)
		if err != nil {
			return err
		}
		srv.TLSConfig = tlsConfig
	}

	// Shutdown fires when ctx cancels (normal path) or when the local
	// cancel below is invoked after ListenAndServe returns (bind failure).
	shutdownCtx, cancel := context.WithCancel(ctx)
	defer cancel()

	shutdownDone := make(chan struct{})
	go func() {
		defer close(shutdownDone)
		<-shutdownCtx.Done()
		graceCtx, cancelGrace := context.WithTimeout(context.Background(), metricsShutdownTimeout)
		defer cancelGrace()
		_ = srv.Shutdown(graceCtx)
	}()

	var err error
	if serveTLS {
		err = srv.ListenAndServeTLS("", "")
	} else {
		err = srv.ListenAndServe()
	}
	cancel()
	<-shutdownDone

	if err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("metrics server: %w", err)
	}
	return nil
}

// metricsTLSConfig loads and reloads the certificate used by the metrics server.
func metricsTLSConfig(ctx context.Context, certPath string) (*tls.Config, error) {
	certFile := filepath.Join(certPath, "tls.crt")
	keyFile := filepath.Join(certPath, "tls.key")
	cert, err := tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return nil, fmt.Errorf("%w: load key pair from cert %q and key %q: %w", errMetricsTLS, certFile, keyFile, err)
	}

	reloader, err := common.NewCertReloader(ctx, certPath, &cert)
	if err != nil {
		return nil, fmt.Errorf("%w: start certificate reloader: %w", errMetricsTLS, err)
	}

	return &tls.Config{
		MinVersion: tls.VersionTLS12,
		GetCertificate: func(*tls.ClientHelloInfo) (*tls.Certificate, error) {
			return reloader.Get(), nil
		},
	}, nil
}
