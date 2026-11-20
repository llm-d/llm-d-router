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
	"sync/atomic"
	"testing"
	"time"

	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	extmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/mocks"
	datasourcemocks "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/mocks"
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

// TestUpdateEndpointRecoversPanickingExtractor verifies that a panicking
// endpoint extractor is recovered, recorded in datalayer_extract_errors_total,
// and neither stops the dispatch loop nor escapes the process.
func TestUpdateEndpointRecoversPanickingExtractor(t *testing.T) {
	metrics.Reset()
	panicking := extmocks.NewEndpointExtractor("panicking").WithExtractPanic("boom")
	working := extmocks.NewEndpointExtractor("working").WithType("mock-endpoint-extractor-working")
	epSrc := notifications.NewEndpointDataSource(notifications.EndpointNotificationSourceType, "ep-source")

	r := NewRuntime(1)
	logger := newTestLogger(t)
	cfg := &Config{
		Sources: []DataSourceConfig{
			{
				Plugin:     epSrc,
				Extractors: []fwkplugin.Plugin{panicking, working},
			},
		},
	}
	require.NoError(t, r.Configure(cfg, logger))

	endpoint := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:      types.NamespacedName{Name: "pod1", Namespace: "default"},
		Address: "1.2.3.4",
	}, nil)

	assert.NotPanics(t, func() { r.UpdateEndpoint(context.Background(), endpoint) },
		"a panicking extractor must not escape the dispatch loop")

	assert.Equal(t, 1.0, promtestutil.ToFloat64(
		metrics.LlmdDataLayerExtractErrorsTotal.WithLabelValues(epSrc.TypedName().Type, panicking.TypedName().Type)),
		"the panic should be recorded as an extract error")

	events := working.GetEvents()
	require.Len(t, events, 1, "the dispatch loop must continue to the next extractor after the panic")
}

// newCrossReplicaRuntime is a runtime with one polling source, polling at the
// default interval, and a cross-replica publisher over an idle syncer.
func newCrossReplicaRuntime(t *testing.T) (*Runtime, *datasourcemocks.MetricsDataSource) {
	t.Helper()
	source := datasourcemocks.NewDataSource(fwkplugin.TypedName{Type: "mock-source", Name: "mock-source"})
	r := NewRuntime(0)
	require.NoError(t, r.Configure(&Config{Sources: []DataSourceConfig{{Plugin: source}}}, newTestLogger(t)))
	r.crossReplicaPub = testCrossReplicaPublisher(&fakeSyncer{})
	return r, source
}

// TestNewEndpointLeavesNoPublisherRegistrationWhenCollectorIsHeld covers a
// NewEndpoint that overlaps a ReleaseEndpoint of the same endpoint: the release
// has deleted the cross-replica registration but not yet removed the collector.
func TestNewEndpointLeavesNoPublisherRegistrationWhenCollectorIsHeld(t *testing.T) {
	r, _ := newCrossReplicaRuntime(t)
	pod := &fwkdl.EndpointMetadata{
		ID:      types.NamespacedName{Name: "pod1", Namespace: "default"},
		Address: "1.2.3.4:5678",
	}
	key := pod.GetID()
	require.True(t, r.collectors.Register(key, NewCollector()), "the release in progress still holds the collector")

	require.Nil(t, r.NewEndpoint(context.Background(), pod), "the endpoint cannot be tracked while its collector is held")
	assert.Empty(t, r.crossReplicaPub.endpointSnapshot(), "a NewEndpoint that returns nil must not leave a cross-replica registration")

	_, removed := r.collectors.Remove(key)
	require.True(t, removed)
	endpoint := r.NewEndpoint(context.Background(), pod)
	require.NotNil(t, endpoint, "the retry after the release completes must track the endpoint")
	r.ReleaseEndpoint(endpoint)
}

// TestNewEndpointStopsItsCollectorWhenThePublisherHoldsTheEndpoint pins the
// rollback of the collector when the cross-replica publisher already holds the
// endpoint: the collector it started must be removed and must stop polling.
func TestNewEndpointStopsItsCollectorWhenThePublisherHoldsTheEndpoint(t *testing.T) {
	r, source := newCrossReplicaRuntime(t)
	pod := &fwkdl.EndpointMetadata{
		ID:      types.NamespacedName{Name: "pod1", Namespace: "default"},
		Address: "1.2.3.4:5678",
	}
	key := pod.GetID()
	require.True(t, r.crossReplicaPub.registerEndpoint(key), "the publisher holds the endpoint")

	require.Nil(t, r.NewEndpoint(context.Background(), pod), "the endpoint cannot be tracked while the publisher holds it")
	polls := atomic.LoadInt64(&source.CallCount)
	// A collector left running polls on every tick of the runtime's interval.
	time.Sleep(3 * r.pollingInterval)
	assert.Equal(t, polls, atomic.LoadInt64(&source.CallCount), "the collector must stop polling once NewEndpoint returns nil")
	_, held := r.collectors.Remove(key)
	assert.False(t, held, "the collector must not stay registered")

	_, err := r.crossReplicaPub.delete(context.Background(), key)
	require.NoError(t, err)
	endpoint := r.NewEndpoint(context.Background(), pod)
	require.NotNil(t, endpoint, "the retry after the publisher releases the endpoint must track it")
	r.ReleaseEndpoint(endpoint)
}
