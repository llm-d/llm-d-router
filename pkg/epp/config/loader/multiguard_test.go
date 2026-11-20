/*
Copyright 2025 The Kubernetes Authors.
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

package loader

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/go-logr/logr"
	configapiv1 "github.com/llm-d/llm-d-router/apix/config/v1"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/saturationdetector/concurrency"
	rollout "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/screener/disaggregatedsetrollout"
)

func TestMultiGuardDependencies(t *testing.T) {
	// Registration matches the EPP runner.
	oldFactory := fwkplugin.Registry[rollout.PluginType]
	oldParser := fwkplugin.PluginsWithPluginDependencies[rollout.PluginType]
	oldGuard := fwkplugin.Registry[concurrency.ConcurrencyDetectorType]
	t.Cleanup(func() {
		if oldFactory == nil {
			delete(fwkplugin.Registry, rollout.PluginType)
		} else {
			fwkplugin.Registry[rollout.PluginType] = oldFactory
		}
		if oldParser == nil {
			delete(fwkplugin.PluginsWithPluginDependencies, rollout.PluginType)
		} else {
			fwkplugin.PluginsWithPluginDependencies[rollout.PluginType] = oldParser
		}
		if oldGuard == nil {
			delete(fwkplugin.Registry, concurrency.ConcurrencyDetectorType)
		} else {
			fwkplugin.Registry[concurrency.ConcurrencyDetectorType] = oldGuard
		}
	})
	fwkplugin.RegisterWithPluginDependencies(rollout.PluginType, fwkplugin.StabilityStable, rollout.Factory, rollout.DependencyParser)
	fwkplugin.Registry[concurrency.ConcurrencyDetectorType] = concurrency.ConcurrencyDetectorFactory
	screen := func(overflow string) configapiv1.PluginSpec {
		return configapiv1.PluginSpec{Name: "screen", Type: rollout.PluginType, Parameters: json.RawMessage(`{"scope":{"labelSelector":"app=test"},"revisionGating":{"mode":"sum","requiredRoles":["prefill","decode"],"overflow":` + overflow + `}}`)}
	}
	guard := func(name string) configapiv1.PluginSpec {
		return configapiv1.PluginSpec{Name: name, Type: concurrency.ConcurrencyDetectorType, Parameters: json.RawMessage(`{"maxConcurrency":8,"failOpen":false}`)}
	}
	multi := screen(`{"guards":[{"role":"prefill","guardPluginName":"p"},{"role":"decode","guardPluginName":"d"}]}`)
	for _, tc := range []struct {
		name  string
		specs []configapiv1.PluginSpec
		fail  bool
	}{
		{"screener first", []configapiv1.PluginSpec{multi, guard("p"), guard("d")}, false},
		{"guards first", []configapiv1.PluginSpec{guard("p"), guard("d"), multi}, false},
		{"missing second", []configapiv1.PluginSpec{multi, guard("p")}, true},
		{"cycle", []configapiv1.PluginSpec{multi, guard("p"), {Name: "d", Type: rollout.PluginType, Parameters: json.RawMessage(`{"revisionGating":{"overflow":{"guardPluginName":"screen"}}}`)}}, true},
		{"legacy", []configapiv1.PluginSpec{screen(`{"guardPluginName":"p"}`), guard("p")}, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := fwkplugin.NewEppHandle(context.Background(), nil)
			_, err := buildPluginDAG(tc.specs, h)
			if (err != nil) != tc.fail {
				t.Fatalf("DAG error=%v, want failure=%v", err, tc.fail)
			}
			if !tc.fail {
				if err := instantiatePlugins(tc.specs, h, logr.Discard()); err != nil {
					t.Fatal(err)
				}
			}
		})
	}
}
