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

/**
 * This file is adapted from Gateway API Inference Extension
 * Original source: https://github.com/kubernetes-sigs/gateway-api-inference-extension/blob/main/cmd/epp/main.go
 * Licensed under the Apache License, Version 2.0
 */

// Package main contains the "Endpoint Picker (EPP)" program for scheduling
// inference requests.
package main

import (
	"os"

	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/llm-d/llm-d-router/cmd/epp/runner"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins"
	"github.com/llm-d/llm-d-router/pkg/metrics"
	"github.com/llm-d/llm-d-router/pkg/telemetry"
)

func main() {
	os.Exit(run())
}

func run() int {
	ctx := ctrl.SetupSignalHandler()

	// Initialize tracing before creating any spans
	shutdownTracing, err := telemetry.InitTracing(ctx)
	if err != nil {
		// Log error but don't fail - tracing is optional
		ctrl.Log.Error(err, "Failed to initialize tracing")
	}
	if shutdownTracing != nil {
		defer func() {
			if err := shutdownTracing(ctx); err != nil {
				ctrl.Log.Error(err, "Failed to shutdown tracing")
			}
		}()
	}

	// Register llm-d-inference-scheduler plugins
	plugins.RegisterAllPlugins()

	// Note: GIE built-in plugins are automatically registered by the runner
	// when it processes configuration in runner.parsePluginsConfiguration()

	if err := runner.NewRunner().
		WithCustomCollectors(metrics.GetCollectors()...).
		Run(ctx); err != nil {
		return 1
	}
	return 0
}
