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

// Package main demonstrates how to build an out-of-tree custom scorer plugin
// and compile it into a custom Endpoint Picker binary and container image.
package main

import (
	"os"

	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/llm-d/llm-d-router/cmd/epp/runner"
	"github.com/llm-d/llm-d-router/examples/customscorer/plugins"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

func init() {
	// Register custom scorer so the EPP runner can instantiate it from config.
	plugin.Register(plugins.Type, plugins.Factory)
}

func main() {
	os.Exit(run())
}

func run() int {
	ctx := ctrl.SetupSignalHandler()
	if err := runner.NewRunner().Run(ctx); err != nil {
		return 1
	}
	return 0
}
