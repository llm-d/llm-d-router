/*
Copyright 2026 Google LLC.

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

// Command exos-epp is the stock llm-d EPP with the Exos EndpointSlice
// discovery plugin registered (out-of-tree registration, see
// docs/plugins-vs-builtins.md#registering-out-of-tree-plugins).
package main

import (
	"os"

	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/llm-d/llm-d-router/cmd/epp/runner"
	"github.com/llm-d/llm-d-router/exos/discovery/endpointslice"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

func main() {
	fwkplugin.Register(endpointslice.PluginType, fwkplugin.StabilityAlpha, endpointslice.Factory)
	if err := runner.NewRunner().Run(ctrl.SetupSignalHandler()); err != nil {
		os.Exit(1)
	}
}
