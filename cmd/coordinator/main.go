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
	"errors"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/spf13/pflag"
	ctrl "sigs.k8s.io/controller-runtime"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

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

func main() {
	configPath := pflag.String("config", "config/coordinator/coordinator.yaml", "path to configuration file")
	metricsPort := pflag.Int("metrics-port", 0, "port for the Prometheus /metrics endpoint (overrides server.metrics_port)")

	logOpts := logutil.NewOptions()
	logOpts.AddFlags(pflag.CommandLine)

	pflag.Parse()

	logutil.InitSetupLogging()
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
	srv, err := server.New(cfg.Server, p)
	if err != nil {
		log.Error(err, "failed to create server")
		os.Exit(1)
	}

	log.Info("starting coordinator", "addr", cfg.Server.ListenAddr, "metrics_port", cfg.Server.MetricsPort)
	log.Info("graceful shutdown enabled", "timeout", cfg.Server.ShutdownTimeout)

	if err := run(srv, cfg.Server); err != nil {
		log.Error(err, "server error")
		os.Exit(1)
	}
}

// run starts the inference server and, alongside it, the Prometheus /metrics
// server. It blocks until either exits or a signal is received; on signal it
// initiates a graceful drain of both bounded by cfg.ShutdownTimeout.
func run(srv *server.Server, cfg config.ServerConfig) error {
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	srvErr := make(chan error, 1)
	go func() { srvErr <- srv.ListenAndServe() }()

	metricsErr := make(chan error, 1)
	go func() { metricsErr <- serveMetrics(ctx, cfg.MetricsPort) }()

	select {
	case err := <-srvErr:
		if !errors.Is(err, http.ErrServerClosed) {
			return err
		}
	case err := <-metricsErr:
		if !errors.Is(err, http.ErrServerClosed) {
			return err
		}
	case <-ctx.Done():
	}
	stop()
	shutdownCtx, cancel := context.WithTimeout(context.Background(), cfg.ShutdownTimeout)
	defer cancel()
	return srv.Shutdown(shutdownCtx)
}

// serveMetrics stands up the Prometheus /metrics endpoint on port and blocks
// until srv exits. Uses the shared controller-runtime registry so any package
// that registers against it (this coordinator's metrics, controller-runtime's
// process collectors) is exposed on the same endpoint.
func serveMetrics(ctx context.Context, port int) error {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(ctrlmetrics.Registry, promhttp.HandlerOpts{EnableOpenMetrics: true}))
	srv := &http.Server{
		Addr:              fmt.Sprintf(":%d", port),
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), metricsShutdownTimeout)
		defer cancel()
		_ = srv.Shutdown(shutdownCtx)
	}()
	if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("metrics server: %w", err)
	}
	return nil
}
