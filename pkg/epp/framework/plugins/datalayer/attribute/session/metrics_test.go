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

package session

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestRegisterAffinityMetrics_Idempotent verifies that two factories
// registering the same collectors against the same registerer succeed
// without error. Both the session-affinity scorer and the session-affinity
// filter call RegisterAffinityMetrics from their factories; in production
// only the second one would otherwise hit AlreadyRegisteredError.
func TestRegisterAffinityMetrics_Idempotent(t *testing.T) {
	registry := prometheus.NewRegistry()
	require.NoError(t, RegisterAffinityMetrics(registry))
	require.NoError(t, RegisterAffinityMetrics(registry))
}

func TestRegisterAffinityMetrics_NilRegisterer(t *testing.T) {
	err := RegisterAffinityMetrics(nil)
	require.Error(t, err)
}

// TestRecordStaleBinding pins the increment plumbing used by the
// session-affinity filter and scorer when their bound endpoint is no
// longer in the candidate set. Removing or relabeling RecordStaleBinding
// in either consumer would silently lose this signal; this test catches
// the case where the recorder itself stops working.
func TestRecordStaleBinding(t *testing.T) {
	const (
		pluginName = "test-record-stale-binding-name"
		pluginType = "test-record-stale-binding-type"
	)
	counter := affinityStaleBindingTotal.WithLabelValues(pluginName, pluginType)
	t.Cleanup(func() { affinityStaleBindingTotal.DeleteLabelValues(pluginName, pluginType) })

	require.Equal(t, 0.0, testutil.ToFloat64(counter), "fresh label set should start at zero")

	RecordStaleBinding(pluginName, pluginType)
	RecordStaleBinding(pluginName, pluginType)
	RecordStaleBinding(pluginName, pluginType)

	assert.Equal(t, 3.0, testutil.ToFloat64(counter))
}
