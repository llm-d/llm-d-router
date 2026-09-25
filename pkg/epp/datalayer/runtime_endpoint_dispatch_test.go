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

package datalayer

import (
	"context"
	"errors"
	"testing"

	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	extmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

// TestNewEndpointDispatchesEventWithNoPollers verifies that endpoint lifecycle
// events are dispatched to EndpointSource even when no PollingDataSource is configured.
// Regression test for: endpoint-notification-source silently drops events when
// no PollingDataSource is present (NewEndpoint returned early before dispatchEndpointEvent).
func TestNewEndpointDispatchesEventWithNoPollers(t *testing.T) {
	extractor := extmocks.NewEndpointExtractor("test-extractor")
	epSrc := notifications.NewEndpointDataSource(notifications.EndpointNotificationSourceType, "ep-source")

	r := NewRuntime(1)
	logger := newTestLogger(t)
	cfg := &Config{
		Sources: []DataSourceConfig{
			{
				Plugin:     epSrc,
				Extractors: []fwkplugin.Plugin{extractor},
			},
		},
	}
	assert.NoError(t, r.Configure(cfg, logger))

	pod := &fwkdl.EndpointMetadata{
		ID:      types.NamespacedName{Name: "pod1", Namespace: "default"},
		Address: "1.2.3.4:5678",
	}

	endpoint := r.NewEndpoint(context.Background(), pod)
	assert.NotNil(t, endpoint, "NewEndpoint should return a valid endpoint")

	events := extractor.GetEvents()
	require.Len(t, events, 1, "EndpointExtractor should receive EventAddOrUpdate from NewEndpoint")
	assert.Equal(t, fwkdl.EventAddOrUpdate, events[0].Type)

	r.ReleaseEndpoint(endpoint)

	events = extractor.GetEvents()
	require.Len(t, events, 2, "EndpointExtractor should receive EventDelete from ReleaseEndpoint")
	assert.Equal(t, fwkdl.EventDelete, events[1].Type)
}

func TestReleaseEndpointContinuesWhenPublisherEndpointIsMissing(t *testing.T) {
	extractor := extmocks.NewEndpointExtractor("test-extractor")
	epSrc := notifications.NewEndpointDataSource(notifications.EndpointNotificationSourceType, "ep-source")

	r := NewRuntime(1)
	logger := newTestLogger(t)
	cfg := &Config{
		Sources: []DataSourceConfig{
			{
				Plugin:     epSrc,
				Extractors: []fwkplugin.Plugin{extractor},
			},
		},
	}
	require.NoError(t, r.Configure(cfg, logger))

	endpoint := r.NewEndpoint(context.Background(), &fwkdl.EndpointMetadata{
		ID:      types.NamespacedName{Name: "pod1", Namespace: "default"},
		Address: "1.2.3.4:5678",
	})
	require.NotNil(t, endpoint)

	r.crossReplicaPub = &crossReplicaPublisher{syncer: &fakeSyncer{}}
	r.ReleaseEndpoint(endpoint)

	events := extractor.GetEvents()
	require.Len(t, events, 2, "publisher miss must not skip endpoint cleanup")
	assert.Equal(t, fwkdl.EventDelete, events[1].Type)
}

func TestUpdateEndpointDispatchesEvent(t *testing.T) {
	extractor := extmocks.NewEndpointExtractor("test-extractor")
	epSrc := notifications.NewEndpointDataSource(notifications.EndpointNotificationSourceType, "ep-source")

	r := NewRuntime(1)
	logger := newTestLogger(t)
	cfg := &Config{
		Sources: []DataSourceConfig{
			{
				Plugin:     epSrc,
				Extractors: []fwkplugin.Plugin{extractor},
			},
		},
	}
	require.NoError(t, r.Configure(cfg, logger))

	endpoint := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:      types.NamespacedName{Name: "pod1", Namespace: "default"},
		Address: "1.2.3.4",
	}, nil)
	endpoint.UpdateMetadata(&fwkdl.EndpointMetadata{
		ID:      types.NamespacedName{Name: "pod1", Namespace: "default"},
		Address: "5.6.7.8",
	})

	r.UpdateEndpoint(context.Background(), endpoint)

	events := extractor.GetEvents()
	require.Len(t, events, 1, "EndpointExtractor should receive EventAddOrUpdate from UpdateEndpoint")
	assert.Equal(t, fwkdl.EventAddOrUpdate, events[0].Type)
	assert.Equal(t, "5.6.7.8", events[0].Endpoint.GetMetadata().Address)
}

// TestUpdateEndpointRecordsExtractErrorMetric verifies that a failing endpoint
// extractor increments datalayer_extract_errors_total with the source and
// extractor type labels, mirroring the http polling path's recording contract.
func TestUpdateEndpointRecordsExtractErrorMetric(t *testing.T) {
	metrics.Reset()
	extractor := extmocks.NewEndpointExtractor("failing-extractor").WithExtractError(errors.New("extract failed"))
	epSrc := notifications.NewEndpointDataSource(notifications.EndpointNotificationSourceType, "ep-source")

	r := NewRuntime(1)
	logger := newTestLogger(t)
	cfg := &Config{
		Sources: []DataSourceConfig{
			{
				Plugin:     epSrc,
				Extractors: []fwkplugin.Plugin{extractor},
			},
		},
	}
	require.NoError(t, r.Configure(cfg, logger))

	endpoint := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:      types.NamespacedName{Name: "pod1", Namespace: "default"},
		Address: "1.2.3.4",
	}, nil)
	r.UpdateEndpoint(context.Background(), endpoint)

	assert.Equal(t, 1.0, promtestutil.ToFloat64(
		metrics.LlmdDataLayerExtractErrorsTotal.WithLabelValues(epSrc.TypedName().Type, extractor.TypedName().Type)),
		"failing endpoint extractor should record one extract error")
}
