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

package weightedfairshare

import (
	"fmt"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"

	metricsutil "github.com/llm-d/llm-d-router/pkg/common/observability/metrics"
)

// Fairness IDs come from a request header, so the flow_id gauges share the
// --fairness-id-metric-label-limit cap with every other fairness_id label.
func TestFlowGauges_RespectFairnessIDLabelLimit(t *testing.T) {
	metricsutil.SetFairnessIDLabelLimit(2)
	t.Cleanup(func() { metricsutil.SetFairnessIDLabelLimit(metricsutil.DefaultFairnessIDLabelLimit) })

	p := newPolicy(t, "")
	for i := range 5 {
		complete(p, fmt.Sprintf("tenant-%d", i), 0, 100, 0)
	}

	assert.Equal(t, 2, testutil.CollectAndCount(p.vserviceGauge))
	assert.Equal(t, 2, testutil.CollectAndCount(p.weightGauge))
	// Accounting still covers every flow; only the series are capped.
	assert.Equal(t, 100.0, virtualService(p, "tenant-4"))
}
