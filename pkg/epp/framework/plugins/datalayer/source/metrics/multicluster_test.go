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

package metrics

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	sourcehttp "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/http"
)

// The factory returns an HTTP metrics source under the distinct multicluster type.
func TestMultiClusterMetricsDataSourceFactory(t *testing.T) {
	p, err := MultiClusterMetricsDataSourceFactory("multicluster-metrics", nil, nil)
	require.NoError(t, err)
	require.IsType(t, &sourcehttp.HTTPDataSource[PrometheusMetricMap]{}, p)
	require.Equal(t, MultiClusterMetricsDataSourceType, p.TypedName().Type)
	require.Equal(t, "multicluster-metrics", p.TypedName().Name)
}

// A malformed interval is rejected at construction rather than silently dropped.
func TestMultiClusterMetricsDataSourceFactoryRejectsBadInterval(t *testing.T) {
	dec := json.NewDecoder(strings.NewReader(`{"interval":"nope"}`))
	_, err := MultiClusterMetricsDataSourceFactory("multicluster-metrics", dec, nil)
	require.Error(t, err)
}

// A valid interval is parsed and reaches the source.
func TestMultiClusterMetricsDataSourceFactoryHonorsInterval(t *testing.T) {
	dec := json.NewDecoder(strings.NewReader(`{"interval":"2s"}`))
	p, err := MultiClusterMetricsDataSourceFactory("multicluster-metrics", dec, nil)
	require.NoError(t, err)
	src := p.(*sourcehttp.HTTPDataSource[PrometheusMetricMap])
	require.Equal(t, 2*time.Second, src.Interval())
}

func TestMultiClusterMetricsDataSourceFactoryFiltersFamilies(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("# TYPE selected gauge\nselected 1\n# TYPE other gauge\nother 2\n"))
	}))
	defer srv.Close()
	u, err := url.Parse(srv.URL)
	require.NoError(t, err)

	dec := json.NewDecoder(strings.NewReader(`{"scheme":"http","families":["selected"]}`))
	p, err := MultiClusterMetricsDataSourceFactory("multicluster-metrics", dec, nil)
	require.NoError(t, err)
	src := p.(*sourcehttp.HTTPDataSource[PrometheusMetricMap])
	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{MetricsHost: u.Host}, nil)
	got, err := src.Poll(context.Background(), ep)
	require.NoError(t, err)
	require.Equal(t, []string{"selected"}, sortedKeys(got))
}
