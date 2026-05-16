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

package datalayer

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	extmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
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
				Extractors: []fwkdl.ExtractorBase{extractor},
			},
		},
	}
	assert.NoError(t, r.Configure(cfg, false, "", logger))

	pod := &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "pod1", Namespace: "default"},
		Address:        "1.2.3.4:5678",
	}

	endpoint := r.NewEndpoint(context.Background(), pod, nil)
	assert.NotNil(t, endpoint, "NewEndpoint should return a valid endpoint")

	events := extractor.GetEvents()
	require.Len(t, events, 1, "EndpointExtractor should receive EventAddOrUpdate from NewEndpoint")
	assert.Equal(t, fwkdl.EventAddOrUpdate, events[0].Type)

	r.ReleaseEndpoint(endpoint)

	events = extractor.GetEvents()
	require.Len(t, events, 2, "EndpointExtractor should receive EventDelete from ReleaseEndpoint")
	assert.Equal(t, fwkdl.EventDelete, events[1].Type)
}
