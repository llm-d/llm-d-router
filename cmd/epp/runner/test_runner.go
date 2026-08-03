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
	"net"

	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"

	"github.com/llm-d/llm-d-router/internal/runnable"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

// NewTestRunnerSetup creates a setup runner dedicated for integration tests. When mockDataSource is
// non-nil, its plugin type is registered as a factory that returns the provided instance, so the
// YAML config can reference it by type name and the runner wires it into the endpoint factory
// automatically. When grpcListener is non-nil the ext_proc server serves on it and
// opts.GRPCPort is ignored.
func NewTestRunnerSetup(ctx context.Context, cfg *rest.Config, opts *runserver.Options, mockDataSource fwkdl.DataSource, grpcListener net.Listener) (*Runner, ctrl.Manager, datastore.Datastore, error) {
	runner := NewRunner()

	if mockDataSource != nil {
		mockType := mockDataSource.TypedName().Type
		fwkplugin.Register(mockType, fwkplugin.StabilityStable, func(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
			return mockDataSource, nil
		})
		defer func() {
			delete(fwkplugin.Registry, mockType)
			delete(fwkplugin.RegistryMetadata, mockType)
		}()
	}

	// Skip controller name validation in integration tests to avoid collisions
	// when multiple controllers are registered within the same test process.
	skipNameValidation := true
	managerOverrides := []func(*ctrl.Options){
		func(o *ctrl.Options) {
			o.Controller.SkipNameValidation = &skipNameValidation
			// The kernel assigns the port, so the bind cannot lose a race to
			// another listener the way a port number chosen in advance can.
			o.Metrics.BindAddress = "127.0.0.1:0"
		},
	}

	manager, ds, err := runner.setup(ctx, cfg, opts, managerOverrides)
	if err != nil {
		return runner, manager, ds, err
	}
	runner.serverRunner.GrpcListener = grpcListener

	// Production runs the ext_proc and health servers on a context that outlives
	// the manager (see Runner.runWithGracefulShutdown). Integration tests drive
	// only mgr.Start, so register them as manager runnables here: they come up
	// with the manager and stop immediately when the test cancels its context,
	// with no drain window.
	if err := manager.Add(runner.serverRunner.AsRunnable(ctrl.Log.WithName("ext-proc"))); err != nil {
		return runner, manager, ds, err
	}
	health := runnable.NoLeaderElection(runnable.GRPCServer("health", runner.healthGRPCServer, runner.healthGRPCPort))
	if err := manager.Add(health); err != nil {
		return runner, manager, ds, err
	}

	return runner, manager, ds, nil
}
