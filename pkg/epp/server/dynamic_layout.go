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

package server

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
)

const (
	// DynamicPDFeatureGate enables runtime P/D/M role reassignment and its admin API.
	DynamicPDFeatureGate    = "dynamicPD"
	DynamicLayoutShowPath   = "/v1/layout/show"
	DynamicLayoutChangePath = "/v1/layout/change"
)

// SetupDynamicLayoutHandlers connects role filters to the dynamic layout admin API.
func SetupDynamicLayoutHandlers(enabled bool, registrar MetricsHandlerRegistrar, plugins fwkplugin.HandlePlugins) error {
	if !enabled {
		return nil
	}
	manager := bylabel.NewLayoutManager()
	hasPrefillFilter := false
	hasDecodeFilter := false
	for _, configuredPlugin := range plugins.GetAllPlugins() {
		if filter, ok := configuredPlugin.(*bylabel.RoleFilter); ok {
			switch filter.TypedName().Type {
			case bylabel.PrefillRoleType:
				hasPrefillFilter = true
			case bylabel.DecodeRoleType:
				hasDecodeFilter = true
			}
		}
	}
	if !hasPrefillFilter || !hasDecodeFilter {
		missing := make([]string, 0, 2)
		if !hasPrefillFilter {
			missing = append(missing, bylabel.PrefillRoleType)
		}
		if !hasDecodeFilter {
			missing = append(missing, bylabel.DecodeRoleType)
		}
		return fmt.Errorf("%s feature gate requires configured role filters: missing %s", DynamicPDFeatureGate, strings.Join(missing, ", "))
	}
	for _, configuredPlugin := range plugins.GetAllPlugins() {
		if filter, ok := configuredPlugin.(*bylabel.RoleFilter); ok {
			filter.WithLayoutManager(manager)
		}
	}
	if err := registrar.AddMetricsServerExtraHandler(DynamicLayoutShowPath, newDynamicLayoutShowHandler(manager)); err != nil {
		return err
	}
	return registrar.AddMetricsServerExtraHandler(DynamicLayoutChangePath, newDynamicLayoutChangeHandler(manager))
}

func writeLayoutJSON(w http.ResponseWriter, status int, value any) {
	payload, err := json.Marshal(value)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to encode layout response: %v", err), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_, _ = w.Write(payload)
}

func newDynamicLayoutShowHandler(manager *bylabel.LayoutManager) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			w.Header().Set("Allow", http.MethodGet)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		state, err := manager.State()
		if err != nil {
			http.Error(w, err.Error(), http.StatusServiceUnavailable)
			return
		}
		writeLayoutJSON(w, http.StatusOK, state)
	})
}

func newDynamicLayoutChangeHandler(manager *bylabel.LayoutManager) http.Handler {
	type request struct {
		Layout string `json:"layout"`
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.Header().Set("Allow", http.MethodPost)
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var body request
		decoder := json.NewDecoder(io.LimitReader(r.Body, 1<<20))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&body); err != nil || body.Layout == "" {
			http.Error(w, "request body must contain a valid layout", http.StatusBadRequest)
			return
		}
		result, err := manager.Apply(body.Layout)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		writeLayoutJSON(w, http.StatusOK, result)
	})
}
