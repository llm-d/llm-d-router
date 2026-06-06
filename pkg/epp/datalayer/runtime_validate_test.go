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

package datalayer

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime/schema"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	extractormocks "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
)

func TestRuntimeConfigureWithNilExtractor(t *testing.T) {
	logger := newTestLogger(t)
	r := NewRuntime(1)

	cfg := &Config{
		Sources: []DataSourceConfig{
			{
				Plugin:     &mocks.MetricsDataSource{},
				Extractors: nil, // nil extractors should be allowed
			},
		},
	}

	err := r.Configure(cfg, logger)
	assert.NoError(t, err, "Configure should succeed with nil extractors")
}

func TestRuntimeConfigureDuplicateGVKFails(t *testing.T) {
	logger := newTestLogger(t)
	r := NewRuntime(1)

	// Two notification sources with the same GVK must fail at Configure.
	gvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}
	src1 := mocks.NewNotificationSource("test", "source1", gvk)
	src2 := mocks.NewNotificationSource("test", "source2", gvk)

	cfg := &Config{
		Sources: []DataSourceConfig{
			{Plugin: src1, Extractors: nil},
			{Plugin: src2, Extractors: nil},
		},
	}

	err := r.Configure(cfg, logger)
	assert.Error(t, err, "Configure should fail with duplicate GVK")
	assert.Contains(t, err.Error(), "duplicate", "Error should mention duplicate GVK")
}

// NotificationSource and EndpointSource dispatch sites type-assert the
// extractor without checking ok. A mismatched extractor type would panic at
// the first event or be silently ignored at dispatch. Configure must catch it.
func TestRuntimeConfigure_NotificationSource_RequiresNotificationExtractor(t *testing.T) {
	logger := newTestLogger(t)
	r := NewRuntime(1)

	gvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}
	src := mocks.NewNotificationSource("notif-src", "notif", gvk)
	// A polling extractor is NOT a NotificationExtractor.
	wrongExt := extractormocks.NewPollingExtractor("polling-ext")

	cfg := &Config{
		Sources: []DataSourceConfig{
			{Plugin: src, Extractors: []fwkplugin.Plugin{wrongExt}},
		},
	}

	err := r.Configure(cfg, logger)
	require.Error(t, err, "Configure must reject a non-NotificationExtractor for a notification source")
	assert.Contains(t, err.Error(), "NotificationExtractor")
}

func TestRuntimeConfigure_EndpointSource_RequiresEndpointExtractor(t *testing.T) {
	logger := newTestLogger(t)
	r := NewRuntime(1)

	src := notifications.NewEndpointDataSource("endpoint-src", "endpoint")
	wrongExt := extractormocks.NewPollingExtractor("polling-ext")

	cfg := &Config{
		Sources: []DataSourceConfig{
			{Plugin: src, Extractors: []fwkplugin.Plugin{wrongExt}},
		},
	}

	err := r.Configure(cfg, logger)
	require.Error(t, err, "Configure must reject a non-EndpointExtractor for an endpoint source")
	assert.Contains(t, err.Error(), "EndpointExtractor")
}

// multiVariantSource implements two variant interfaces. The registerSource
// type-switch routes by first match; without an explicit guard the source
// would silently land in whichever case comes first.
type multiVariantSource struct {
	typedName fwkplugin.TypedName
	gvk       schema.GroupVersionKind
}

func (m *multiVariantSource) TypedName() fwkplugin.TypedName                     { return m.typedName }
func (m *multiVariantSource) GVK() schema.GroupVersionKind                       { return m.gvk }
func (m *multiVariantSource) Dispatch(_ context.Context, _ fwkdl.Endpoint) error { return nil }
func (m *multiVariantSource) AppendExtractor(_ fwkplugin.Plugin) error           { return nil }
func (m *multiVariantSource) Notify(_ context.Context, event fwkdl.NotificationEvent) (*fwkdl.NotificationEvent, error) {
	// Body never runs in this test. Configure rejects the source for
	// implementing multiple variants before any Notify call. Echo the
	// event so the return satisfies nilnil.
	return &event, nil
}

func TestRuntimeConfigure_SourceImplementingMultipleVariants_Rejected(t *testing.T) {
	logger := newTestLogger(t)
	r := NewRuntime(1)

	src := &multiVariantSource{
		typedName: fwkplugin.TypedName{Type: "ambiguous", Name: "ambiguous"},
		gvk:       schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
	}

	cfg := &Config{
		Sources: []DataSourceConfig{{Plugin: src}},
	}

	err := r.Configure(cfg, logger)
	require.Error(t, err, "a source that implements multiple variant interfaces must be rejected")
	assert.Contains(t, err.Error(), "multiple variant")
}
