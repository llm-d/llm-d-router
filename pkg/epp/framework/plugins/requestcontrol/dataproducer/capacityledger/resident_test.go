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

package capacityledger

import (
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

var testStart = time.Date(2026, 9, 29, 12, 0, 0, 0, time.UTC)

// newTestRequest returns a request with a tokenized prompt of the given length.
func newTestRequest(prompt int, stream bool, maxOutput *int64) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		RequestID: "req",
		Body: &fwkrh.InferenceRequestBody{
			TokenizedRequest: &fwkrh.TokenizedRequest{Prompts: []fwkrh.PromptTokens{{TokenIDs: make([]uint32, prompt)}}},
			Stream:           stream,
			MaxOutputTokens:  maxOutput,
		},
	}
}

func TestFootprint(t *testing.T) {
	g := geometry{blockSize: 16, stepBudget: 2048, decodeStep: 1}
	tests := []struct {
		name string
		r    residentState
		g    geometry
		want vec
	}{
		{name: "prefilling", r: residentState{prompts: []int64{100}, uncachedPrompt: 52, n: 1}, g: g,
			want: vec{7, 52, 1}},
		{name: "decoding", r: residentState{prompts: []int64{100}, uncachedPrompt: 52, n: 1, age: 20, decoding: true}, g: g,
			want: vec{8, 1, 1}},
		{name: "step capped at the budget", r: residentState{prompts: []int64{5000}, uncachedPrompt: 5000, n: 1}, g: g,
			want: vec{313, 2048, 1}},
		{name: "n sequences", r: residentState{prompts: []int64{100}, uncachedPrompt: 100, n: 4}, g: g,
			want: vec{28, 400, 4}},
		{name: "n sequences decoding with speculative tokens", r: residentState{prompts: []int64{100}, n: 4, decoding: true},
			g: geometry{blockSize: 16, stepBudget: 2048, decodeStep: 3}, want: vec{28, 12, 4}},
		{name: "two prompts", r: residentState{prompts: []int64{100, 50}, uncachedPrompt: 150, n: 1}, g: g,
			want: vec{11, 150, 2}},
		{name: "block size unknown", r: residentState{prompts: []int64{100}, uncachedPrompt: 100, n: 1},
			g: geometry{stepBudget: 2048, decodeStep: 1}, want: vec{0, 100, 1}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			require.Equal(t, tc.want, footprint(tc.r, tc.g))
		})
	}
}

func TestSequencesPerPrompt(t *testing.T) {
	req := newTestRequest(100, true, nil)
	require.Equal(t, int64(1), sequencesPerPrompt(req, 8))
	req.Body.Payload = fwkrh.PayloadMap{"n": float64(4)}
	require.Equal(t, int64(4), sequencesPerPrompt(req, 8))
	req.Body.Payload = fwkrh.PayloadMap{"n": "4"}
	require.Equal(t, int64(1), sequencesPerPrompt(req, 8), "a malformed n is ignored")
	req.Body.Payload = fwkrh.PayloadMap{"n": float64(1 << 62)}
	require.Equal(t, int64(8), sequencesPerPrompt(req, 8), "n is clamped to the limit")
}

func TestPromptLengthsAndUncachedTokens(t *testing.T) {
	req := newTestRequest(100, true, nil)
	lengths := promptLengths(req, 500)
	require.Equal(t, []int64{100}, lengths)
	require.Equal(t, int64(100), sum(lengths))
	require.Equal(t, []int64{250}, promptLengths(&fwksched.InferenceRequest{}, 250))
	require.Equal(t, int64(1), sequencesPerPrompt(req, 16))

	attrs := fwkdl.NewAttributes()
	attrs.Put(attrprefix.PrefixCacheMatchInfoDataKey, attrprefix.NewPrefixCacheMatchInfo(1, 6, 16).WithCachedBlockCount(3))
	ep := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{}, &fwkdl.Metrics{}, attrs)
	require.Equal(t, int64(100-3*16), uncachedPromptTokens(ep, 100, attrprefix.PrefixCacheMatchInfoDataKey))
}
