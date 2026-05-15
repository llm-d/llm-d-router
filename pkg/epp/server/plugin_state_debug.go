/*
Copyright 2026 The Kubernetes Authors.

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

package server

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"

	ctrl "sigs.k8s.io/controller-runtime"

	fwkplugin "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
)

const PluginStateDebugPath = "/debug/plugins/state"

type pluginStateDebugResponse struct {
	Plugins []pluginStateDebugEntry `json:"plugins"`
}

type pluginStateDebugEntry struct {
	Name  string `json:"name"`
	Type  string `json:"type"`
	State any    `json:"state"`
}

func SetupPluginStateDebugHandler(mgr ctrl.Manager, plugins fwkplugin.HandlePlugins) error {
	if plugins == nil {
		return errors.New("plugin handle is not configured")
	}
	return mgr.AddMetricsServerExtraHandler(PluginStateDebugPath, NewPluginStateDebugHandler(plugins))
}

func NewPluginStateDebugHandler(plugins fwkplugin.HandlePlugins) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		if plugins == nil {
			http.Error(w, "plugin handle is not configured", http.StatusInternalServerError)
			return
		}

		payload, err := json.Marshal(collectPluginState(plugins))
		if err != nil {
			http.Error(w, fmt.Sprintf("failed to encode plugin state: %v", err), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write(payload)
	})
}

func collectPluginState(plugins fwkplugin.HandlePlugins) pluginStateDebugResponse {
	allPlugins := plugins.GetAllPluginsWithNames()
	names := make([]string, 0, len(allPlugins))
	for name := range allPlugins {
		names = append(names, name)
	}
	sort.Strings(names)

	response := pluginStateDebugResponse{
		Plugins: make([]pluginStateDebugEntry, 0, len(names)),
	}
	for _, name := range names {
		plugin := allPlugins[name]
		if plugin == nil {
			continue
		}
		dumper, ok := plugin.(fwkplugin.StateDumper)
		if !ok {
			continue
		}

		response.Plugins = append(response.Plugins, pluginStateDebugEntry{
			Name:  name,
			Type:  plugin.TypedName().Type,
			State: dumper.DumpState(),
		})
	}

	return response
}
