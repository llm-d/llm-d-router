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

package runner

import (
	"context"
	"encoding/json"

	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"

	backendmetrics "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/backend/metrics"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/datastore"
	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	runserver "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/server"
)

// NewTestRunnerSetup creates a setup runner dedicated for integration tests. When mockDataSource is
// non-nil, its plugin type is registered as a factory that returns the provided instance, so the
// YAML config can reference it by type name and the runner wires it into the endpoint factory
// automatically. Pass nil to fall back to the legacy metrics system with pmc.
func NewTestRunnerSetup(ctx context.Context, cfg *rest.Config, opts *runserver.Options, pmc backendmetrics.PodMetricsClient, mockDataSource fwkdl.DataSource) (ctrl.Manager, datastore.Datastore, error) {
	runner := NewRunner()

	if mockDataSource != nil {
		mockType := mockDataSource.TypedName().Type
		fwkplugin.Register(mockType, func(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
			return mockDataSource, nil
		})
		defer delete(fwkplugin.Registry, mockType)
	}

	// Skip controller name validation in integration tests to avoid collisions
	// when multiple controllers are registered within the same test process.
	skipNameValidation := true
	managerOverrides := []func(*ctrl.Options){
		func(o *ctrl.Options) {
			o.Controller.SkipNameValidation = &skipNameValidation
		},
	}

	return runner.setup(ctx, cfg, opts, pmc, managerOverrides)
}
