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

package groupsched

// The loud block-size failure must survive the capacity simplification: a request landing on an
// engine whose measured block size disagrees with the configured one FAILS the request.

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

func TestPreRequestStillFailsOnBlockSizeMismatch(t *testing.T) {
	h := newGoldenHarness(t, goldenPodsFor(2, 20))
	h.setBlockSize("e1", 32) // e1 disagrees with the configured 64
	h.learn()
	require.False(t, h.r.sched.HasCapacityFor("golden/e1"),
		"a mismatched engine must not hold capacity: its budget would be in the wrong unit")

	req := h.request(1, 300)
	h.identify(req, "T1", 1)
	h.enqueue(req)
	h.saturate(true)
	for _, d := range h.pickLoop() {
		_ = d
	}

	res := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{goldenEndpoint("e1", 20, 32)}},
		},
	}
	err := h.r.PreRequest(context.Background(), req, res)
	require.Error(t, err, "a PreRequest on a mismatched engine must fail the request")
	require.Contains(t, err.Error(), "CacheBlockSize", "the failure must name the block-size disagreement")
	_ = time.Second
}
