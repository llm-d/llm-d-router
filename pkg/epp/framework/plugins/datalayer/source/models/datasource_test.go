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
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
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
