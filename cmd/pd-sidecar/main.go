/*
Copyright 2025 The llm-d Authors.

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
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
	"github.com/spf13/pflag"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	sidecarmetrics "github.com/llm-d/llm-d-router/pkg/sidecar/metrics"
	"github.com/llm-d/llm-d-router/pkg/sidecar/proxy"
	"github.com/llm-d/llm-d-router/pkg/sidecar/version"
)

// metricsShutdownTimeout bounds graceful shutdown of the metrics server so a
// scraper holding a connection at process exit cannot block termination.
const metricsShutdownTimeout = 5 * time.Second

// serveMetrics starts a standalone Prometheus metrics HTTP server serving
// controller-runtime's registry, where sidecar metrics are registered.
func serveMetrics(ctx context.Context, port int) error {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.HandlerFor(ctrlmetrics.Registry, promhttp.HandlerOpts{EnableOpenMetrics: true}))
	srv := &http.Server{Addr: fmt.Sprintf(":%d", port), Handler: mux}
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

func main() {
	// Initialize options with defaults
	opts := proxy.NewOptions()

	// Add options flags (including logging flags)
	opts.AddFlags(pflag.CommandLine)
	pflag.Parse()

	logger := opts.NewLogger()
	log.SetLogger(logger)

	ctx := ctrl.SetupSignalHandler()
	log.IntoContext(ctx, logger)

	// Complete options (handles migration from deprecated flags, populates Config)
	if err := opts.Complete(); err != nil {
		logger.Error(err, "Failed to complete configuration")
		return
	}

	// Validate options
	if err := opts.Validate(); err != nil {
		logger.Error(err, "Invalid configuration")
		return
	}

	// Initialize tracing conditionally using config
	if opts.Tracing {
		shutdown, err := tracing.InitTracing(ctx, logger, "llm-d-disagg-sidecar")
		if err != nil {
			// Log error but don't fail - tracing is optional
			logger.Error(err, "Failed to initialize tracing")
		} else if shutdown != nil {
			defer func() {
				if err := shutdown(context.Background()); err != nil {
					logger.Error(err, "Failed to shutdown tracing")
				}
			}()
		}
	}

	if opts.MetricsPort > 0 {
		sidecarmetrics.Register()
		go func() {
			logger.Info("starting metrics server", "port", opts.MetricsPort)
			if err := serveMetrics(ctx, opts.MetricsPort); err != nil {
				logger.Error(err, "metrics server failed")
			}
		}()
	}

	logger.Info("Proxy starting", "Built on", version.BuildRef, "From Git SHA", version.CommitSHA)
	logger.Info("Proxy configuration", "config", opts.Config)

	proxyServer := proxy.NewProxy(opts.Config)
	if err := proxyServer.Start(ctx); err != nil {
		logger.Error(err, "failed to start proxy server")
	}
}
