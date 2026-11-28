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

package inflightload

import (
	"context"
	"testing"

	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

func TestKVEventPrefillCompletion_RepeatedLastBlockFallsBack(t *testing.T) {
	cases := []struct {
		name        string
		promptLen   int
		repeatAt    int
		sharedStamp bool
		generated   bool
		fullPrompt  bool
	}{
		{name: "unique_last_block", promptLen: 256, repeatAt: -1},
		{name: "repeated_first_block", promptLen: 256, repeatAt: 0},
		{name: "repeated_middle_block", promptLen: 256, repeatAt: 64},
		{name: "partial_prompt_block", promptLen: 264, repeatAt: 0},
		{name: "shared_session", promptLen: 256, repeatAt: 0, sharedStamp: true},
		{name: "generated_block", promptLen: 256, repeatAt: 0, generated: true},
		{name: "shared_session_generated_block", promptLen: 256, repeatAt: 0, sharedStamp: true, generated: true},
		{name: "full_prompt_repeated_block", promptLen: 256, repeatAt: 0, fullPrompt: true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			name := "kv-block-position-" + tc.name
			producer := newKVEventsProducer(t, name)
			ctx := t.Context()
			prefill := newEventEndpoint("prefill-0", bylabel.RolePrefill, prefillAddress)
			decode := newEventEndpoint("decode-0", bylabel.RoleDecode, decodeAddress)
			prompt := seq(0, tc.promptLen)
			end := len(prompt) / blockSize * blockSize
			if tc.repeatAt >= 0 {
				copy(prompt[end-blockSize:end], prompt[tc.repeatAt:tc.repeatAt+blockSize])
			}
			req := newStampedRequest("req-block-position", prompt)
			stamp := req.RequestID
			if tc.sharedStamp {
				stamp = "client-session"
				req.Headers[sessionIDHeader] = stamp
			}
			res := pdResult(prefill, decode)
			req.SchedulingResult = res
			require.NoError(t, producer.PreRequest(ctx, req, res))
			t.Cleanup(func() {
				producer.ResponseBody(context.Background(), req, &fwkrh.Response{EndOfStream: true}, nil)
			})
			gauge := inflightTokens.WithLabelValues("prefill-0", "default", name, "", "0")
			require.Equal(t, float64(tc.promptLen), promtestutil.ToFloat64(gauge))

			source := kvevents.EventSource{Endpoint: decodeSource}
			firstEnd := max(tc.repeatAt+blockSize, blockSize)
			require.NoError(t, producer.ProcessEvents(ctx, source, storeBatch(stamp, prompt[:firstEnd])))
			require.Equal(t, float64(tc.promptLen), promtestutil.ToFloat64(gauge),
				"an early occurrence of the final block must retain prefill load")
			require.Equal(t, float64(1), pendingWatch(name))
			require.Equal(t, int64(1), producer.GetRequests(fullEndpointName("prefill-0")))

			stored := prompt[end-blockSize : end]
			if tc.fullPrompt {
				stored = prompt[:end]
			}
			require.NoError(t, producer.ProcessEvents(ctx, source, storeBatch(stamp, stored)))
			if tc.repeatAt >= 0 && !tc.fullPrompt {
				require.Equal(t, float64(tc.promptLen), promtestutil.ToFloat64(gauge),
					"identical block contents cannot establish their prompt position")
				require.Equal(t, float64(1), pendingWatch(name))
				if tc.generated {
					require.NoError(t, producer.ProcessEvents(ctx, source, storeBatch(stamp, seq(9000, blockSize))))
					if tc.sharedStamp {
						require.Equal(t, float64(tc.promptLen), promtestutil.ToFloat64(gauge))
					} else {
						require.Zero(t, promtestutil.ToFloat64(gauge))
						require.Equal(t, float64(1), completions(name, signalKVEvent))
					}
				}
			} else {
				require.Zero(t, promtestutil.ToFloat64(gauge))
				require.Zero(t, pendingWatch(name))
				require.Equal(t, float64(1), completions(name, signalKVEvent))
			}
			producer.ResponseBody(ctx, req, &fwkrh.Response{StartOfStream: true}, nil)
			require.Zero(t, promtestutil.ToFloat64(gauge))
			require.Zero(t, pendingWatch(name))
		})
	}
}
