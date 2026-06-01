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

package mocks

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"time"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

var (
	_ fwkdl.PollingDispatcher  = (*MetricsDataSource)(nil)
	_ fwkdl.DataSource         = (*NotificationSource)(nil)
	_ fwkdl.NotificationSource = (*NotificationSource)(nil)
)

// MetricsDataSource is a test PollingDispatcher: on Dispatch it writes the
// preconfigured metrics into the endpoint and returns a sentinel error. The
// sentinel signals "test dispatch ran" without producing real Poll data.
type MetricsDataSource struct {
	mu        sync.RWMutex
	typedName plugin.TypedName
	CallCount int64
	metrics   map[types.NamespacedName]*fwkdl.Metrics
	errors    map[types.NamespacedName]error
}

func NewDataSource(typedName plugin.TypedName) *MetricsDataSource {
	return &MetricsDataSource{typedName: typedName}
}

func (fds *MetricsDataSource) TypedName() plugin.TypedName {
	return fds.typedName
}

// SetMetrics replaces the metrics map in a thread-safe manner.
func (fds *MetricsDataSource) SetMetrics(metrics map[types.NamespacedName]*fwkdl.Metrics) {
	fds.mu.Lock()
	defer fds.mu.Unlock()
	fds.metrics = metrics
}

// SetErrors replaces the errors map in a thread-safe manner.
func (fds *MetricsDataSource) SetErrors(errors map[types.NamespacedName]error) {
	fds.mu.Lock()
	defer fds.mu.Unlock()
	fds.errors = errors
}

// Dispatch satisfies PollingDispatcher. Preserves the original mock side
// effect (write metrics into the endpoint when keyed by NamespacedName) and
// returns the sentinel error so tests can detect the dispatch ran.
func (fds *MetricsDataSource) Dispatch(_ context.Context, ep fwkdl.Endpoint) error {
	atomic.AddInt64(&fds.CallCount, 1)
	fds.mu.RLock()
	defer fds.mu.RUnlock()
	nn := ep.GetMetadata().Clone().NamespacedName
	if metrics, ok := fds.metrics[nn]; ok {
		if _, ok := fds.errors[nn]; !ok {
			clone := metrics.Clone()
			clone.UpdateTime = time.Now()
			ep.UpdateMetrics(clone)
		}
	}
	return errors.New("sentinel nothing polled")
}

// AppendExtractor is a no-op: this mock sets endpoint metrics directly in
// Dispatch instead of running extractors.
func (fds *MetricsDataSource) AppendExtractor(_ plugin.Plugin) error { return nil }

// NotificationSource implements DataSource + NotificationSource for testing.
type NotificationSource struct {
	typedName plugin.TypedName
	gvk       schema.GroupVersionKind
}

func NewNotificationSource(pluginType, name string, gvk schema.GroupVersionKind) *NotificationSource {
	return &NotificationSource{
		typedName: plugin.TypedName{Type: pluginType, Name: name},
		gvk:       gvk,
	}
}

func (m *NotificationSource) TypedName() plugin.TypedName  { return m.typedName }
func (m *NotificationSource) GVK() schema.GroupVersionKind { return m.gvk }
func (m *NotificationSource) Extractors() []string         { return []string{} }
func (m *NotificationSource) Collect(_ context.Context, _ fwkdl.Endpoint) error {
	return nil
}

func (m *NotificationSource) Notify(_ context.Context, event fwkdl.NotificationEvent) (*fwkdl.NotificationEvent, error) {
	return &event, nil
}
