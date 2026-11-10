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

package runner

import (
	"context"
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
	fwknet "github.com/llm-d/llm-d-router/test/framework/net"
)

// --fairness-id-metric-label-limit caps the fairness_id label in file-discovery
// mode as it does on the Kubernetes path.
func TestRunWithFileDiscovery_AppliesFairnessIDLabelLimit(t *testing.T) {
	t.Cleanup(func() { metricsutil.SetFairnessIDLabelLimit(metricsutil.DefaultFairnessIDLabelLimit) })

	modelServer, _ := newDrainTestBackend(t, &atomic.Bool{})
	opts := newDrainTestOptions(modelServer)
	opts.FairnessIDMetricLabelLimit = 1
	plugin, discovery := drainTestFileDiscoveryConfig(t, modelServer)
	opts.ConfigText = fmt.Sprintf(`apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
plugins:
%s- type: random-picker
schedulingProfiles:
- name: default
  plugins:
  - pluginRef: random-picker
dataLayer:
%s`, plugin, discovery)

	ctx, stop := context.WithCancel(t.Context())
	defer stop()
	listener, err := fwknet.ReserveListener()
	require.NoError(t, err)
	t.Cleanup(func() { _ = listener.Close() })
	startFileDiscoveryDrainTest(ctx, t, opts, listener)

	// A served request means startup has finished applying the flags. It carries no
	// fairness ID, so it resolves to default-flow and takes no cap slot.
	_, routed := sendDrainTestRequest(t, listener, "ready")
	select {
	case res := <-routed:
		require.NoError(t, res.err)
	case <-time.After(10 * time.Second):
		t.Fatal("EPP did not serve a request")
	}

	require.Equal(t, "first", metricsutil.BoundFairnessID("first"))
	require.Equal(t, metricsutil.OverflowValue, metricsutil.BoundFairnessID("second"))
}
