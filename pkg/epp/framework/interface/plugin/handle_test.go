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

package plugin_test

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

func TestNewEppHandleMetrics(t *testing.T) {
	recorder := prometheus.NewRegistry()
	handle := plugin.NewEppHandle(t.Context(), func() []types.NamespacedName { return nil }, plugin.WithMetricsRecorder(recorder))

	require.Same(t, recorder, handle.Metrics())
}

func TestNewEppHandleMetricsDefaultsToNil(t *testing.T) {
	handle := plugin.NewEppHandle(t.Context(), nil)

	require.Nil(t, handle.Metrics())
	require.Nil(t, handle.PodList())
}
