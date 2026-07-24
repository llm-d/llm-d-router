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

package metrics

import (
	"bytes"
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/http"
)

func TestMetricsDataSourceFactory_TLS(t *testing.T) {
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
			ds, err := MetricsDataSourceFactory("m", json.NewDecoder(bytes.NewBufferString(tt.params)), nil)
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
	_, err := http.NewHTTPDataSource("invalid", "/metrics", http.TLSOptions{SkipVerify: true}, MetricsDataSourceType,
		"metrics-data-source", parseMetrics)
	assert.NotNil(t, err, "expected to fail with invalid scheme")

	source, err := http.NewHTTPDataSource("https", "/metrics", http.TLSOptions{SkipVerify: true}, MetricsDataSourceType,
		"metrics-data-source", parseMetrics)
	assert.Nil(t, err, "failed to create HTTP datasource")

	dsType := source.TypedName().Type
	assert.Equal(t, MetricsDataSourceType, dsType)

	ctx := context.Background()
	endpoint := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{
			Name:      "pod1",
			Namespace: "default",
		},
		Address: "1.2.3.4:5678",
	}, nil)
	_, err = source.Poll(ctx, endpoint)
	assert.NotNil(t, err, "expected to fail polling for metrics")
}
