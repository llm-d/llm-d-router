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

// Package models
package models

import (
	"context"
	"encoding/json"
	nethttp "net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/models"
	extmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/models"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/http"
)

func TestModelDataSourceFactory_TLS(t *testing.T) {
	tests := []struct {
		name    string
		params  string
		wantErr error
	}{
		{name: "https no certs", params: `{"scheme":"https"}`},
		{name: "client cert wired to loader", params: `{"scheme":"https","clientCertPath":"/nope/c.pem","clientKeyPath":"/nope/k.pem"}`, wantErr: http.ErrLoadClientCert},
		{name: "ca path wired to loader", params: `{"scheme":"https","insecureSkipVerify":false,"caCertPath":"/nope/ca.pem"}`, wantErr: http.ErrReadCACert},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ds, err := ModelDataSourceFactory("m", fwkplugin.StrictDecoder(json.RawMessage(tt.params)), nil)
			if tt.wantErr != nil {
				assert.ErrorIs(t, err, tt.wantErr)
				return
			}
			assert.NoError(t, err)
			assert.NotNil(t, ds)
		})
	}
}

func TestDatasource(t *testing.T) {
	srcPlugin, err := ModelDataSourceFactory("models-data-source",
		fwkplugin.StrictDecoder(json.RawMessage(`{"scheme":"https","path":"/models","insecureSkipVerify":true}`)), nil)
	assert.Nil(t, err, "failed to create http datasource")
	source := srcPlugin.(fwkdl.PollingDispatcher)

	extPlugin, err := extmodels.ModelServerExtractorFactory("models-data-extractor", nil, nil)
	assert.Nil(t, err, "failed to create extractor")

	cfg := &datalayer.Config{
		Sources: []datalayer.DataSourceConfig{
			{
				Plugin:     source,
				Extractors: []fwkplugin.Plugin{extPlugin},
			},
		},
	}

	pollingInterval := 50 * time.Millisecond
	runtime := datalayer.NewRuntime(pollingInterval)

	err = runtime.Configure(cfg, logr.Logger{})
	assert.Nil(t, err, "failed to configure runtime")

	ctx := context.Background()
	pod := &fwkdl.EndpointMetadata{
		ID: types.NamespacedName{
			Name:      "pod1",
			Namespace: "default",
		},
		Address: "1.2.3.4:5678",
	}

	endpoint := runtime.NewEndpoint(ctx, pod)
	assert.NotNil(t, endpoint, "failed to create endpoint")

	err = source.Dispatch(ctx, endpoint)
	assert.NotNil(t, err, "expected dispatch to fail (no real HTTP target)")
}

func TestModelsExtractorBinding(t *testing.T) {
	server := httptest.NewServer(nethttp.HandlerFunc(func(w nethttp.ResponseWriter, _ *nethttp.Request) {
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"llama"},{"id":"sql-v3","parent":"llama"}]}`))
	}))
	t.Cleanup(server.Close)
	host := strings.TrimPrefix(server.URL, "http://")

	tests := []struct {
		name        string
		sources     []string // models-data-source names
		listedUnder string   // source that lists the extractor in config, "" for none
		wantBound   []string // sources whose poll reaches the extractor
	}{
		{name: "listed under the source", sources: []string{"a"}, listedUnder: "a", wantBound: []string{"a"}},
		{name: "declared but not listed", sources: []string{"a"}, wantBound: []string{"a"}},
		{name: "listed under one of several sources", sources: []string{"a", "b"}, listedUnder: "a", wantBound: []string{"a"}},
		{name: "not listed with several sources", sources: []string{"a", "b"}},
		{name: "no models-data-source"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			extractor := extmodels.NewModelExtractor()
			runtime := datalayer.NewRuntime(50 * time.Millisecond)
			require.NoError(t, extractor.RegisterDependencies(runtime))

			sources := make(map[string]fwkdl.PollingDispatcher, len(test.sources))
			cfg := &datalayer.Config{}
			for _, name := range test.sources {
				source, err := NewHTTPModelsDataSource("http", "/v1/models", name)
				require.NoError(t, err)
				sources[name] = source
				srcCfg := datalayer.DataSourceConfig{Plugin: source}
				if name == test.listedUnder {
					srcCfg.Extractors = []fwkplugin.Plugin{extractor}
				}
				cfg.Sources = append(cfg.Sources, srcCfg)
			}

			require.NoError(t, runtime.Configure(cfg, logr.Discard()))

			for name, source := range sources {
				endpoint := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{MetricsHost: host}, nil)
				require.NoError(t, source.Dispatch(t.Context(), endpoint))
				_, ok := fwkdl.ReadAttribute[attrmodels.ModelDataCollection](endpoint.GetAttributes(), attrmodels.ModelsAttributeKey)
				assert.Equal(t, slices.Contains(test.wantBound, name), ok, "extractor bound to source %s", name)
			}
		})
	}
}
