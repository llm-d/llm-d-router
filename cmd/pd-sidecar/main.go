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
	"github.com/spf13/pflag"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/sidecar/proxy"
	"github.com/llm-d/llm-d-router/pkg/sidecar/version"
	"github.com/llm-d/llm-d-router/pkg/telemetry"
)

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

	// Initialize tracing before creating any spans
	shutdownTracing, err := telemetry.InitTracing(ctx)
	if err != nil {
		// Log error but don't fail - tracing is optional
		logger.Error(err, "Failed to initialize tracing")
	}
	if shutdownTracing != nil {
		defer func() {
			if err := shutdownTracing(ctx); err != nil {
				logger.Error(err, "Failed to shutdown tracing")
			}
		}()
	}

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

	logger.Info("Proxy starting", "Built on", version.BuildRef, "From Git SHA", version.CommitSHA)
	logger.Info("Proxy configuration", "config", opts.Config)

	proxyServer := proxy.NewProxy(opts.Config)
	if err := proxyServer.Start(ctx); err != nil {
		logger.Error(err, "failed to start proxy server")
	}
}
