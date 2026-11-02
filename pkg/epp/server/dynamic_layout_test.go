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
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
)

func TestDynamicLayoutShowReturnsUnavailableBeforeEndpointObservation(t *testing.T) {
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, DynamicLayoutShowPath, nil)

	newDynamicLayoutShowHandler(bylabel.NewLayoutManager()).ServeHTTP(recorder, request)

	require.Equal(t, http.StatusServiceUnavailable, recorder.Code)
	require.Contains(t, recorder.Body.String(), bylabel.TensorParallelSizeLabel)
}

func TestDynamicLayoutChangeRejectsInvalidRequest(t *testing.T) {
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, DynamicLayoutChangePath, strings.NewReader(`{"unknown":"value"}`))

	newDynamicLayoutChangeHandler(bylabel.NewLayoutManager()).ServeHTTP(recorder, request)

	require.Equal(t, http.StatusBadRequest, recorder.Code)
}

func TestDynamicLayoutHandlersRejectWrongMethods(t *testing.T) {
	manager := bylabel.NewLayoutManager()
	for _, test := range []struct {
		handler http.Handler
		method  string
		path    string
		allow   string
	}{
		{newDynamicLayoutShowHandler(manager), http.MethodPost, DynamicLayoutShowPath, http.MethodGet},
		{newDynamicLayoutChangeHandler(manager), http.MethodGet, DynamicLayoutChangePath, http.MethodPost},
	} {
		recorder := httptest.NewRecorder()
		test.handler.ServeHTTP(recorder, httptest.NewRequest(test.method, test.path, nil))
		require.Equal(t, http.StatusMethodNotAllowed, recorder.Code)
		require.Equal(t, test.allow, recorder.Header().Get("Allow"))
	}
}

func TestDynamicLayoutSetupIsNoOpWhenDisabled(t *testing.T) {
	require.NoError(t, SetupDynamicLayoutHandlers(false, nil, nil))
}

func TestDynamicLayoutSetupRequiresBothRoleFilters(t *testing.T) {
	for _, test := range []struct {
		name        string
		plugins     map[string]fwkplugin.Plugin
		wantMissing string
	}{
		{name: "both missing", plugins: map[string]fwkplugin.Plugin{}, wantMissing: "prefill-filter, decode-filter"},
		{name: "prefill missing", plugins: map[string]fwkplugin.Plugin{"decode-filter": bylabel.NewDecodeRole()}, wantMissing: "prefill-filter"},
		{name: "decode missing", plugins: map[string]fwkplugin.Plugin{"prefill-filter": bylabel.NewPrefillRole()}, wantMissing: "decode-filter"},
	} {
		t.Run(test.name, func(t *testing.T) {
			handle := fwkplugin.NewEppHandle(context.Background(), nil)
			for name, configuredPlugin := range test.plugins {
				handle.AddPlugin(name, configuredPlugin)
			}

			err := SetupDynamicLayoutHandlers(true, nil, handle)

			require.EqualError(t, err, "dynamicPD feature gate requires configured role filters: missing "+test.wantMissing)
		})
	}
}
