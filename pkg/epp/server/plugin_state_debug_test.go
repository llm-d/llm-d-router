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

package server

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
)

type stateDebugTestPlugin struct {
	typedName fwkplugin.TypedName
}

func (p *stateDebugTestPlugin) TypedName() fwkplugin.TypedName {
	return p.typedName
}

type stateDebugTestDumper struct {
	stateDebugTestPlugin
	state any
}

func (p *stateDebugTestDumper) DumpState() any {
	return p.state
}

func withFrozenNow(t *testing.T, fixed time.Time) {
	t.Helper()
	prev := nowFunc
	nowFunc = func() time.Time { return fixed }
	t.Cleanup(func() { nowFunc = prev })
}

func TestPluginStateDebugHandlerIncludesDumpers(t *testing.T) {
	withFrozenNow(t, time.Date(2025, 1, 2, 3, 4, 5, 0, time.UTC))

	handle := fwkplugin.NewEppHandle(context.Background(), nil)
	handle.AddPlugin("skip", &stateDebugTestPlugin{
		typedName: fwkplugin.TypedName{Type: "skip-type", Name: "skip"},
	})
	handle.AddPlugin("z-dumper", &stateDebugTestDumper{
		stateDebugTestPlugin: stateDebugTestPlugin{typedName: fwkplugin.TypedName{Type: "test-type", Name: "z-dumper"}},
		state:                map[string]any{"count": 2},
	})
	handle.AddPlugin("a-dumper", &stateDebugTestDumper{
		stateDebugTestPlugin: stateDebugTestPlugin{typedName: fwkplugin.TypedName{Type: "test-type", Name: "a-dumper"}},
		state:                map[string]any{"value": "first"},
	})

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, PluginStateDebugPath, nil)
	NewPluginStateDebugHandler(handle).ServeHTTP(recorder, request)

	require.Equal(t, http.StatusOK, recorder.Code)
	require.Equal(t, "application/json", recorder.Header().Get("Content-Type"))
	require.JSONEq(t, `{
		"timestamp": "2025-01-02T03:04:05Z",
		"plugins": {
			"a-dumper": {"type":"test-type","state":{"value":"first"}},
			"z-dumper": {"type":"test-type","state":{"count":2}}
		}
	}`, recorder.Body.String())

	var response pluginStateDebugResponse
	require.NoError(t, json.Unmarshal(recorder.Body.Bytes(), &response))
	require.Len(t, response.Plugins, 2)
	require.Contains(t, response.Plugins, "a-dumper")
	require.Contains(t, response.Plugins, "z-dumper")
	require.NotContains(t, response.Plugins, "skip")
}

func TestPluginStateDebugHandlerRejectsNonGet(t *testing.T) {
	handle := fwkplugin.NewEppHandle(context.Background(), nil)
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, PluginStateDebugPath, nil)

	NewPluginStateDebugHandler(handle).ServeHTTP(recorder, request)

	require.Equal(t, http.StatusMethodNotAllowed, recorder.Code)
	require.Equal(t, http.MethodGet, recorder.Header().Get("Allow"))
}

func TestPluginStateDebugHandlerReportsEncodingErrors(t *testing.T) {
	handle := fwkplugin.NewEppHandle(context.Background(), nil)
	handle.AddPlugin("bad-dumper", &stateDebugTestDumper{
		stateDebugTestPlugin: stateDebugTestPlugin{typedName: fwkplugin.TypedName{Type: "test-type", Name: "bad-dumper"}},
		state:                func() {},
	})

	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodGet, PluginStateDebugPath, nil)
	NewPluginStateDebugHandler(handle).ServeHTTP(recorder, request)

	require.Equal(t, http.StatusInternalServerError, recorder.Code)
	require.Contains(t, recorder.Body.String(), "failed to encode plugin state")
}
