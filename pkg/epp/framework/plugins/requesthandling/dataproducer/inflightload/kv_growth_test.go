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
	"math"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// The response chunks of a kvGrowth test arrive at chunkTime. The usage is scraped after them at
// scrapeTime, a refresh interval later, and at readTime.
var (
	chunkTime  = time.Unix(1_700_000_000, 0)
	scrapeTime = chunkTime.Add(time.Millisecond)
	readTime   = scrapeTime.Add(2 * kvGrowthRefreshInterval)
)

// projectedAfterScrapes returns the endpoint's projection at readTime. The scrapes before that one
// began after the response chunks, so it shows the prompts and the streamed tokens of the requests.
func projectedAfterScrapes(k *kvGrowth, endpointID string) int64 {
	k.projected(endpointID, scrapeTime, scrapeTime)
	k.projected(endpointID, scrapeTime.Add(kvGrowthRefreshInterval), scrapeTime.Add(kvGrowthRefreshInterval))
	return k.projected(endpointID, readTime, readTime)
}

// growthRequest is one dispatched request in a kvGrowth test: prompt tokens, cached of them on the
// endpoint. With done > 0 it has ended at done tokens, cut short when cut is set. Otherwise it is
// running on endpoint "a" after streamed events, or has not sent a response chunk when queued is
// set. It generates choices sequences for each of its prompts; prompt, streamed and done count the
// tokens of all of them.
type growthRequest struct {
	prompt, cached, outputCap int64
	prompts, choices          int64 // 0 stands for 1
	streamed                  int
	done                      int64
	cut, queued               bool
}

func TestKVGrowth_Projected(t *testing.T) {
	const prompt = 8
	tests := []struct {
		name                string
		z                   float64
		maxOutputTokens     int64
		minWindow, minPrior int
		requests            []growthRequest
		want                int64
	}{
		{
			// No request has completed, so every sample runs to the cap: two running requests grow
			// by 4 each and the next request takes its 15-token prompt plus 4.
			name: "before any completion each request is reserved its output cap",
			z:    3, maxOutputTokens: 4, minWindow: kvGrowthMinWindow, minPrior: kvGrowthMinPrior,
			requests: []growthRequest{{prompt: 10, outputCap: 4}, {prompt: 20, outputCap: 4}},
			want:     27,
		},
		{
			// The request has not sent a response chunk, so its 8-token prompt is not in the
			// scraped usage yet: 8 on top of its 4 output tokens and 8 + 4 for the next request.
			name: "a request without a response chunk also reserves its prompt",
			z:    3, maxOutputTokens: 4, minWindow: kvGrowthMinWindow, minPrior: kvGrowthMinPrior,
			requests: []growthRequest{{prompt: prompt, outputCap: 4, queued: true}},
			want:     24,
		},
		{
			// Five of the 8 prompt tokens are cached on the endpoint. vLLM counts cached blocks as
			// free until a request takes them, so the request reserves all 8 and the next request's
			// prompt is taken as 8 as well.
			name: "a request without a response chunk reserves the cached part of its prompt too",
			z:    3, maxOutputTokens: 4, minWindow: kvGrowthMinWindow, minPrior: kvGrowthMinPrior,
			requests: []growthRequest{{prompt: prompt, cached: 5, outputCap: 4, queued: true}},
			want:     24,
		},
		{
			// Completions at 1 and at the cap of 8 give S = [1, 1/2, ..., 1/2]. The running request
			// frees only the 2 prompt tokens that were not cached, whatever shares the rest: eight
			// tokens ahead it holds 10 with probability 1/2, a mean change of 5 - 2, and the next
			// request holds 16 with probability 1/2.
			name: "a request frees the part of its prompt that was not cached",
			z:    0, maxOutputTokens: 8, minWindow: 3, minPrior: 0,
			requests: []growthRequest{{prompt: prompt, done: 1}, {prompt: prompt, done: 8}, {prompt: prompt, cached: 6}},
			want:     11,
		},
		{
			// The request generates two sequences of up to 4 tokens: 2 * 4. The next request is taken
			// to do the same, with a copy of its prompt for each choice: 2 * 8 + 2 * 4.
			name: "a request for several choices reserves the output of each",
			z:    3, maxOutputTokens: 4, minWindow: kvGrowthMinWindow, minPrior: kvGrowthMinPrior,
			requests: []growthRequest{{prompt: prompt, outputCap: 4, choices: 2}},
			want:     32,
		},
		{
			// Before its first response chunk the request also reserves a prompt for each choice.
			name: "a request for several choices reserves a prompt for each",
			z:    3, maxOutputTokens: 4, minWindow: kvGrowthMinWindow, minPrior: kvGrowthMinPrior,
			requests: []growthRequest{{prompt: prompt, outputCap: 4, choices: 2, queued: true}},
			want:     48,
		},
		{
			// Two prompts of 8 tokens in total are two sequences of up to 4 tokens. The request has
			// not sent a response chunk: 8 + 2 * 4 for it, and the same for the next request.
			name: "a request with several prompts reserves the output of a sequence for each",
			z:    3, maxOutputTokens: 4, minWindow: kvGrowthMinWindow, minPrior: kvGrowthMinPrior,
			requests: []growthRequest{{prompt: prompt, outputCap: 4, prompts: 2, queued: true}},
			want:     32,
		},
		{
			// Two choices for each of three prompts are six sequences: 6 * 4 for the request and
			// 2 * 8 + 6 * 4 for the next one.
			name: "choices are generated for each prompt",
			z:    3, maxOutputTokens: 4, minWindow: kvGrowthMinWindow, minPrior: kvGrowthMinPrior,
			requests: []growthRequest{{prompt: prompt, outputCap: 4, prompts: 3, choices: 2}},
			want:     64,
		},
		{
			// Four streamed tokens are two on each sequence, which leaves six of 8 for each. Six
			// tokens ahead the request has grown by 2 * 6 and the next request holds 2 * 8 + 2 * 6.
			name: "streamed tokens are shared between the sequences",
			z:    3, maxOutputTokens: 8, minWindow: kvGrowthMinWindow, minPrior: kvGrowthMinPrior,
			requests: []growthRequest{{prompt: prompt, choices: 2, streamed: 4}},
			want:     40,
		},
		{
			// The ended request generated two sequences, 4 tokens in total, and the length of each
			// is not known: S stays 1. The sampled requests average 12 prompt tokens and 1.5
			// sequences, so the running request grows by 4 and the next one takes 12 + 1.5 * 4.
			name: "a request with several sequences is not an observed length",
			z:    0, maxOutputTokens: 4, minWindow: 2, minPrior: 0,
			requests: []growthRequest{{prompt: prompt, outputCap: 4, choices: 2, done: 4}, {prompt: prompt, outputCap: 4}},
			want:     22,
		},
		{
			// The running request has streamed 4 tokens over four sequences, 1 on each, so it has not
			// outlived the two completions at 2: S = [1, 1, 0, 0]. The sampled requests average 16
			// prompt tokens and two sequences. One token ahead the running request has grown by 4
			// and the next one holds 16 + 2; after that both have ended.
			name: "a running request with several sequences is sampled at the length of one",
			z:    3, maxOutputTokens: 4, minWindow: 4, minPrior: 0,
			requests: []growthRequest{
				{prompt: prompt, outputCap: 4, done: 2}, {prompt: prompt, outputCap: 4, done: 2},
				{prompt: prompt, outputCap: 4, choices: 4, streamed: 4},
			},
			want: 22,
		},
		{
			// Both sequences of the running request are at their limit of 1, so it ends and frees its
			// 10 tokens. Until then a next request with two choices holds 2 * 8 prompt tokens.
			name: "the next request's prompt is reserved for each of its choices",
			z:    3, maxOutputTokens: 4, minWindow: 4, minPrior: 0,
			requests: []growthRequest{{prompt: prompt, outputCap: 1, choices: 2, streamed: 2}},
			want:     16,
		},
		{
			// 16-token steps: the running request grows by 4096 and the next one takes 100 + 4096.
			name: "the steps cover the configured output cap",
			z:    3, maxOutputTokens: 4096, minWindow: kvGrowthMinWindow, minPrior: kvGrowthMinPrior,
			requests: []growthRequest{{prompt: 100, outputCap: 4096}},
			want:     8292,
		},
		{
			// A completion at 10 tokens. The projection steps 16 tokens at a time: one step ahead
			// the running request and the next one may each have grown by 16, and after that both
			// have ended.
			name: "the projection steps 16 tokens at a time for a 4096-token cap",
			z:    3, maxOutputTokens: 4096, minWindow: 2, minPrior: 0,
			requests: []growthRequest{{prompt: 100, outputCap: 4096, done: 10}, {prompt: 100, outputCap: 4096}},
			want:     132,
		},
		{
			// Same sample. One 16-token step ahead the running request can only have grown by its
			// limit of 1, and the next request takes 100 + 16.
			name: "a request is not charged past its own output limit",
			z:    3, maxOutputTokens: 4096, minWindow: 2, minPrior: 0,
			requests: []growthRequest{{prompt: 100, outputCap: 4096, done: 10}, {prompt: 100, outputCap: 1}},
			want:     117,
		},
		{
			// Three completions at 1 token and a request running at 3. The steps are 4 tokens, but
			// the request has outlived the completions: it is taken to run to the cap, 1021 more
			// tokens, where the next request holds 8 + 1024 with probability 1/4.
			name: "a request is conditioned on the lengths it has outlived within a step",
			z:    0, maxOutputTokens: 1024, minWindow: 4, minPrior: 0,
			requests: []growthRequest{
				{prompt: prompt, done: 1}, {prompt: prompt, done: 1}, {prompt: prompt, done: 1},
				{prompt: prompt, streamed: 3},
			},
			want: 1279,
		},
		{
			// Steps of 16 tokens, lengths kept to the token. Three completions at 2 tokens and a
			// request running at 1, which has not outlived them: one step ahead it may have grown by
			// 16 and the next request holds 8 + 16, and after that both have ended.
			name: "a request has not outlived the longer lengths within its step",
			z:    0, maxOutputTokens: 4096, minWindow: 4, minPrior: 0,
			requests: []growthRequest{
				{prompt: prompt, done: 2}, {prompt: prompt, done: 2}, {prompt: prompt, done: 2},
				{prompt: prompt, streamed: 1},
			},
			want: 40,
		},
		{
			// Two completions at 20 tokens and a request running at 10. One 16-token step ahead it
			// would be past 20, so it has ended by then: the peak is the first step, 16 + 8 + 16.
			name: "survival ahead is read from the request's own length",
			z:    0, maxOutputTokens: 4096, minWindow: 3, minPrior: 0,
			requests: []growthRequest{{prompt: prompt, done: 20}, {prompt: prompt, done: 20}, {prompt: prompt, streamed: 10}},
			want:     40,
		},
		{
			// Steps of 32 tokens, lengths kept to 2 tokens. Three completions at 3, one at 35, and a
			// request running at 3 that counts as having outlived the first three. One step ahead it
			// has streamed 35 at most and is still running; two steps ahead it has ended. The next
			// request holds 8 + 64 one step ahead with probability 2/5: 64 + 28.8.
			name: "lengths are kept at a sixteenth of a step",
			z:    0, maxOutputTokens: 8192, minWindow: 5, minPrior: 0,
			requests: []growthRequest{
				{prompt: prompt, done: 3}, {prompt: prompt, done: 3}, {prompt: prompt, done: 3},
				{prompt: prompt, done: 35}, {prompt: prompt, streamed: 3},
			},
			want: 93,
		},
		{
			// Twenty one-token completions and the twenty assumed requests give S = [1, 1/2, 1/2, 1/2].
			// Four tokens ahead the mean is -2 + 6 and the variance 2 * (1/4) * 12^2 = 72, so with
			// z = 1 the peak is 4 + sqrt(72) = 12.49.
			name: "the assumed requests stay in the sample after completions",
			z:    1, maxOutputTokens: 4, minWindow: kvGrowthMinWindow, minPrior: kvGrowthMinPrior,
			requests: append(slices.Repeat([]growthRequest{{prompt: prompt, outputCap: 4, done: 1}}, 20),
				growthRequest{prompt: prompt, outputCap: 4}),
			want: 12,
		},
		{
			// Same sample. With z = 3 the mean plus the margin is 4 + 3 * sqrt(72) = 29.46, more than
			// the requests can take: the running one has 4 tokens left and the next takes 8 + 4.
			name: "the margin does not exceed what the requests can still take",
			z:    3, maxOutputTokens: 4, minWindow: kvGrowthMinWindow, minPrior: kvGrowthMinPrior,
			requests: append(slices.Repeat([]growthRequest{{prompt: prompt, outputCap: 4, done: 1}}, 20),
				growthRequest{prompt: prompt, outputCap: 4}),
			want: 16,
		},
		{
			// A completion at 1 and the running request give S = [1, 1/2, 1/2, 1/2]; the ended
			// request with three sequences is not an observed length. The sampled requests average
			// 16 prompt tokens and two sequences. Four tokens ahead the mean plus the margin is
			// 12 + 3 * sqrt(208), more than the requests can take: the running one has 2 * 4 tokens
			// left and the next takes 16 + 2 * 4.
			name: "what the requests can still take counts every sequence",
			z:    3, maxOutputTokens: 4, minWindow: 4, minPrior: 0,
			requests: []growthRequest{
				{prompt: prompt, outputCap: 4, done: 1}, {prompt: prompt, outputCap: 4, choices: 3, done: 12},
				{prompt: prompt, outputCap: 4, choices: 2},
			},
			want: 32,
		},
		{
			// Thirty one-token completions and a running request make a sample of 31, which gets
			// three assumed requests: S = [1, 1/11, 1/11, 1/11]. Four tokens ahead the mean is
			// 2 * 4/11 and the variance 2 * (10/121) * 16, so the peak is 0.73 + 3 * 1.63 = 5.6.
			name: "one assumed request is added for every ten sampled requests",
			z:    3, maxOutputTokens: 4, minWindow: 40, minPrior: 2,
			requests: append(slices.Repeat([]growthRequest{{outputCap: 4, done: 1}}, 30), growthRequest{outputCap: 4}),
			want:     6,
		},
		{
			// Completions at 1, 1 and 3 give S = [1, 1/3, 1/3, 0]. The mean peaks at the first
			// token: 1 for the running request plus 9 for the next one.
			name: "mean of the learned distribution",
			z:    0, maxOutputTokens: 4, minWindow: 4, minPrior: 0,
			requests: []growthRequest{
				{prompt: prompt, outputCap: 4, done: 1}, {prompt: prompt, outputCap: 4, done: 1},
				{prompt: prompt, outputCap: 4, done: 3}, {prompt: prompt, outputCap: 4},
			},
			want: 10,
		},
		{
			// Same sample. Three tokens ahead the mean is -13/3 + 11/3 and the variance is
			// 2 * (1/3)(2/3) * 11^2 = (22/3)^2, so with z = 2 the peak is -2/3 + 2 * 22/3 = 14.
			name: "z standard deviations are added",
			z:    2, maxOutputTokens: 4, minWindow: 4, minPrior: 0,
			requests: []growthRequest{
				{prompt: prompt, outputCap: 4, done: 1}, {prompt: prompt, outputCap: 4, done: 1},
				{prompt: prompt, outputCap: 4, done: 3}, {prompt: prompt, outputCap: 4},
			},
			want: 14,
		},
		{
			// S = [1, 1/2, 1/2, 1/4]. The running request can emit one more token and then frees
			// its 11 tokens, so three tokens ahead the peak is -11 + 5.5 + 3 * 5.5.
			name: "streamed tokens lower the reserve",
			z:    3, maxOutputTokens: 4, minWindow: 4, minPrior: 0,
			requests: []growthRequest{
				{prompt: prompt, outputCap: 4, done: 1}, {prompt: prompt, outputCap: 4, done: 1},
				{prompt: prompt, outputCap: 4, done: 3}, {prompt: prompt, outputCap: 4, streamed: 3},
			},
			want: 11,
		},
		{
			// The running request stops after one token, so three tokens ahead the peak is
			// -8 + 11/3 + 3 * sqrt((2/9) * 121) = 11.22.
			name: "a request stops at its own output cap",
			z:    3, maxOutputTokens: 4, minWindow: 4, minPrior: 0,
			requests: []growthRequest{
				{prompt: prompt, outputCap: 4, done: 1}, {prompt: prompt, outputCap: 4, done: 1},
				{prompt: prompt, outputCap: 4, done: 3}, {prompt: prompt, outputCap: 1},
			},
			want: 11,
		},
		{
			// A limit of 9 above the configured cap of 4 counts as 4: the running request has two
			// tokens left and the peak, two tokens ahead, is 2 + 8 + 2.
			name: "a request's output limit above the configured cap is ignored",
			z:    3, maxOutputTokens: 4, minWindow: kvGrowthMinWindow, minPrior: kvGrowthMinPrior,
			requests: []growthRequest{{prompt: prompt, outputCap: 9, streamed: 2}},
			want:     12,
		},
		{
			// A request cut off by its own cap of 2 says nothing about how long a request capped at
			// 4 runs: S stays 1, the running request grows by 4 and the next takes 8 + 4.
			name: "a completion at the request's own cap is not an observed length",
			z:    3, maxOutputTokens: 4, minWindow: 4, minPrior: 0,
			requests: []growthRequest{{prompt: prompt, outputCap: 2, done: 2}, {prompt: prompt, outputCap: 4}},
			want:     16,
		},
		{
			// A stream that ended at 2 tokens without completing would have run longer, so S stays 1
			// as above.
			name: "a stream cut short is not an observed length",
			z:    3, maxOutputTokens: 4, minWindow: 4, minPrior: 0,
			requests: []growthRequest{{prompt: prompt, outputCap: 4, done: 2, cut: true}, {prompt: prompt, outputCap: 4}},
			want:     16,
		},
		{
			// Only the completion at 1 is in the window, so S = [1, 0, 0, 0] and the peak is the
			// first token: 1 + 9. With the completion at 3 in the sample it would be 26.
			name: "the sample is the most recent requests",
			z:    3, maxOutputTokens: 4, minWindow: 2, minPrior: 0,
			requests: []growthRequest{
				{prompt: prompt, outputCap: 4, done: 3}, {prompt: prompt, outputCap: 4, done: 1},
				{prompt: prompt, outputCap: 4},
			},
			want: 10,
		},
		{
			// Two requests are running, so the two completions dispatched after them stay in the
			// sample: S = [1, 2/3, 2/3, 0]. Three tokens ahead the first request has ended and the
			// second and the next one each hold 11 with probability 2/3. With z = 2.5:
			// -10 - 2/3 + 22/3 + 2.5 * 22/3 = 15.
			name: "the sample grows with the number of running requests",
			z:    2.5, maxOutputTokens: 4, minWindow: 2, minPrior: 0,
			requests: []growthRequest{
				{prompt: prompt, outputCap: 4, streamed: 2}, {prompt: prompt, outputCap: 4},
				{prompt: prompt, outputCap: 4, done: 3}, {prompt: prompt, outputCap: 4, done: 1},
			},
			want: 15,
		},
		{
			// A completion at 2 and a request still running at 2 give S = [1, 1, 1/2, 1/2]: half of
			// the requests that reach two tokens go on. Four tokens ahead the next request adds
			// 6 + 3 * 6 and the running one has freed its 10.
			name: "a completion counts before a request still running at the same length",
			z:    3, maxOutputTokens: 4, minWindow: 4, minPrior: 0,
			requests: []growthRequest{{prompt: prompt, outputCap: 4, done: 2}, {prompt: prompt, outputCap: 4, streamed: 2}},
			want:     14,
		},
		{
			// The running request has reached its cap of 1 and frees 9 tokens when it ends, more than
			// the next request adds in any later step. Until then the next request's 8-token prompt
			// sits next to it.
			name: "the next request's prompt is reserved while a finishing request still holds its tokens",
			z:    3, maxOutputTokens: 4, minWindow: 4, minPrior: 0,
			requests: []growthRequest{{prompt: prompt, outputCap: 1, streamed: 1}},
			want:     8,
		},
		{
			// Five later completions push the running request out of the sample: S = [1, 0, 0, 0]
			// says nothing about a request already at 2 tokens. It grows by 2 more, and one token
			// ahead the next request adds 9.
			name: "a running request that has outlived the sample is assumed to run to its cap",
			z:    0, maxOutputTokens: 4, minWindow: 1, minPrior: 0,
			requests: append([]growthRequest{{prompt: prompt, outputCap: 4, streamed: 2}},
				slices.Repeat([]growthRequest{{prompt: prompt, outputCap: 4, done: 1}}, 5)...),
			want: 10,
		},
		{
			// Nine stream events against a configured cap of 4: the model server does not stop
			// requests there, so the request is taken to keep its tokens. The next request takes
			// 8 + 4.
			name: "a request that streams past the configured cap keeps its tokens",
			z:    3, maxOutputTokens: 4, minWindow: kvGrowthMinWindow, minPrior: kvGrowthMinPrior,
			requests: []growthRequest{{prompt: prompt, outputCap: 4, streamed: 9}},
			want:     12,
		},
		{
			name: "an endpoint with no running request reserves nothing",
			z:    3, maxOutputTokens: 4, minWindow: kvGrowthMinWindow, minPrior: kvGrowthMinPrior,
			requests: []growthRequest{{prompt: prompt, outputCap: 4, done: 2}},
			want:     0,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			k := newKVGrowth(tc.z, tc.maxOutputTokens)
			k.minWindow, k.minPrior = tc.minWindow, tc.minPrior
			for i, r := range tc.requests {
				id := strconv.Itoa(i)
				k.add("a", id, r.prompt, r.prompt-r.cached, r.outputCap, r.prompts, r.choices)
				if !r.queued {
					k.progress(id, r.streamed, chunkTime)
				}
				if r.done > 0 {
					k.finish(id, r.done, !r.cut)
				}
			}
			require.Equal(t, tc.want, projectedAfterScrapes(k, "a"))
			require.Zero(t, projectedAfterScrapes(k, "b"), "requests on another endpoint are not counted")
		})
	}
}

// A stream that completes without a usage block is not a known length: a model server can send
// several tokens in one stream event. It counts as longer than the most tokens it streamed.
func TestKVGrowth_CompletionWithoutUsageIsLongerThanItStreamed(t *testing.T) {
	k := newKVGrowth(0, 8)
	k.minWindow, k.minPrior = 3, 0
	// The requests carry no prompt, so the projection is their output alone.
	k.add("a", "no-usage", 0, 0, 8, 1, 1)
	k.progress("no-usage", 3, chunkTime)
	k.progress("no-usage", 1, chunkTime)
	k.finish("no-usage", 0, true)
	k.progress("no-usage", 5, chunkTime) // a chunk after the end of the request is not recorded
	k.add("a", "usage", 0, 0, 8, 1, 1)
	k.progress("usage", 2, chunkTime)
	k.finish("usage", 2, true)
	k.add("a", "running", 0, 0, 8, 1, 1)
	k.progress("running", 0, chunkTime)

	// At 2 tokens two requests were still running and one of them ended; the stream without usage
	// is not known to have ended at 3. S = [1, 1, .5, .5, .5, .5, .5, .5]: eight tokens ahead the
	// running request and the next one each hold 8 tokens with a chance of one half.
	require.Equal(t, int64(8), projectedAfterScrapes(k, "a"))
}

// A request that ends without output, such as an error status the model server delivers in full,
// carries no length and leaves the sample: it does not push the completion at 2 tokens out of the
// two-request window, and its 100-token prompt does not count toward the next request's.
func TestKVGrowth_RequestWithoutOutputLeavesTheSample(t *testing.T) {
	k := newKVGrowth(0, 4)
	k.minWindow, k.minPrior = 2, 0
	k.add("a", "completed", 8, 8, 4, 1, 1)
	k.progress("completed", 2, chunkTime)
	k.finish("completed", 2, true)
	k.add("a", "failed", 100, 100, 4, 1, 1)
	k.progress("failed", 0, chunkTime)
	k.finish("failed", 0, true)
	k.add("a", "running", 8, 8, 4, 1, 1)
	k.progress("running", 0, chunkTime)

	// S = [1, 1, 0, 0]: two tokens ahead the running request has grown by 2 and the next adds 10.
	require.Equal(t, int64(12), projectedAfterScrapes(k, "a"))
}

// The projection stays in range for any configured cap.
func TestKVGrowth_ProjectionSaturates(t *testing.T) {
	k := newKVGrowth(3, math.MaxInt64)
	k.add("a", "r", 8, 8, 0, 1, 1)
	require.Equal(t, int64(kvGrowthMaxProjection), projectedAfterScrapes(k, "a"))
}

func TestKVGrowth_RequestIsAddedOnce(t *testing.T) {
	k := newKVGrowth(3, 4)
	k.add("a", "r", 8, 8, 4, 1, 1)
	k.add("b", "r", 8, 8, 4, 1, 1)
	k.progress("r", 0, chunkTime)
	require.Equal(t, int64(4+12), projectedAfterScrapes(k, "a"))
	require.Zero(t, projectedAfterScrapes(k, "b"))

	k.finish("r", 0, true)
	require.Zero(t, projectedAfterScrapes(k, "a"))
}

func TestKVGrowth_ProjectionIsReusedUntilTheEndpointChanges(t *testing.T) {
	k := newKVGrowth(3, 4)
	k.add("a", "r", 8, 8, 4, 1, 1)
	k.progress("r", 0, chunkTime)
	require.Equal(t, int64(4+12), projectedAfterScrapes(k, "a"))

	k.progress("r", 3, chunkTime)
	k.add("b", "other", 8, 8, 4, 1, 1)
	require.Equal(t, int64(4+12), k.projected("a", readTime.Add(kvGrowthRefreshInterval/2), readTime),
		"reused within the interval, whatever is dispatched to other endpoints")
	// The running request has one token left: 1 for it plus 8 + 1 for the next request, and the
	// 3 tokens it streamed since the scrape.
	later := readTime.Add(2 * kvGrowthRefreshInterval)
	require.Equal(t, int64(3+1+9), k.projected("a", later, readTime))

	// The new request's 8-token prompt is not in the scraped usage, and one token ahead it adds 1
	// as well.
	k.add("a", "r2", 8, 8, 4, 1, 1)
	require.Equal(t, int64(3+8+11), k.projected("a", later, readTime), "a dispatch to the endpoint is counted at once")

	// With r gone, r2 grows by 4 and the next request takes 8 + 4.
	k.finish("r", 4, true)
	require.Equal(t, int64(8+4+12), k.projected("a", later, readTime), "a completion on the endpoint is counted at once")
}

// A projection computed late in the life of the survival table expires with the table.
func TestKVGrowth_ProjectionIsNotReusedPastItsSurvivalTable(t *testing.T) {
	k := newKVGrowth(0, 4)
	k.minWindow, k.minPrior = 4, 0
	k.add("a", "r", 8, 8, 4, 1, 1)
	k.progress("r", 0, chunkTime)
	k.add("b", "other", 8, 8, 4, 1, 1)
	k.progress("other", 0, chunkTime)
	require.Equal(t, int64(4+12), projectedAfterScrapes(k, "a"))

	// The completion on b is not in the table yet: r and r2 are reserved 4 each, r2 its prompt.
	k.finish("other", 1, true)
	k.add("a", "r2", 8, 8, 4, 1, 1)
	require.Equal(t, int64(8+4+4+12), k.projected("a", readTime.Add(kvGrowthRefreshInterval-time.Millisecond), readTime))

	// S = [1, 0, 0, 0] once the table is rebuilt: one token ahead r and r2 have grown by 1 and
	// the next request adds 9.
	require.Equal(t, int64(8+2+9), k.projected("a", readTime.Add(kvGrowthRefreshInterval+50*time.Millisecond), readTime))
}

// The engine has taken a request's prompt when the first response chunk arrives. A scrape that
// completes after the chunk may have read the usage before it, so the prompt stays reserved until
// the scrape after that one.
func TestKVGrowth_PromptStaysReservedUntilAScrapeThatBeganAfterTheFirstChunk(t *testing.T) {
	k := newKVGrowth(3, 4)
	k.add("a", "r", 8, 8, 4, 1, 1)
	k.progress("r", 0, chunkTime)
	k.progress("r", 0, chunkTime.Add(time.Hour)) // a later chunk does not count as the first one

	// Every read is a refresh interval after the one before, so each is recomputed. Reserved are
	// 8 for the prompt, 4 for the output still to come and 8 + 4 for the next request.
	read := chunkTime
	for _, tc := range []struct {
		name      string
		scrapedAt time.Time
		want      int64
	}{
		{"scraped before the chunk", chunkTime.Add(-time.Millisecond), 8 + 4 + 12},
		{"scraped with the chunk", chunkTime, 8 + 4 + 12},
		{"the first scrape after the chunk", scrapeTime, 8 + 4 + 12},
		{"the same scrape read again", scrapeTime, 8 + 4 + 12},
		{"the scrape after it", scrapeTime.Add(time.Millisecond), 4 + 12},
	} {
		read = read.Add(kvGrowthRefreshInterval)
		require.Equal(t, tc.want, k.projected("a", read, tc.scrapedAt), tc.name)
	}
}

// Tokens a request streams after a scrape are not in the scraped usage. The projection has seen a
// scrape when the tokens may already have been streamed past what it read, so they stay reserved
// until the second scrape after that one.
func TestKVGrowth_StreamedTokensStayReservedUntilTheScrapedUsageIncludesThem(t *testing.T) {
	k := newKVGrowth(3, 8)
	k.add("a", "r", 8, 8, 8, 1, 1)
	k.progress("r", 0, chunkTime)
	// 8 output tokens still to come and 8 + 8 for the next request.
	require.Equal(t, int64(8+16), projectedAfterScrapes(k, "a"))

	k.progress("r", 3, chunkTime)
	// Five tokens are still to come, and five tokens ahead the next request holds 8 + 5. Every read
	// is a refresh interval after the one before.
	read := readTime
	for _, tc := range []struct {
		name      string
		scrapedAt time.Time
		want      int64
	}{
		{"the usage has not been scraped since", readTime, 3 + 5 + 13},
		{"the first scrape seen after the tokens", readTime.Add(time.Millisecond), 3 + 5 + 13},
		{"the scrape after it", readTime.Add(2 * time.Millisecond), 3 + 5 + 13},
		{"the second scrape after it", readTime.Add(3 * time.Millisecond), 5 + 13},
	} {
		read = read.Add(kvGrowthRefreshInterval)
		require.Equal(t, tc.want, k.projected("a", read, tc.scrapedAt), tc.name)
	}
}

// Response chunks are recorded while a projection is being computed: every chunk of every stream
// reports its progress, so progress does not wait for the lock the projections hold.
func TestKVGrowth_ProgressDoesNotWaitForAProjection(t *testing.T) {
	k := newKVGrowth(3, 4)
	k.add("a", "r", 8, 8, 4, 1, 1)

	k.mu.Lock()
	recorded := make(chan struct{})
	go func() {
		k.progress("r", 2, chunkTime) // the first chunk
		k.progress("r", 3, chunkTime)
		close(recorded)
	}()
	select {
	case <-recorded:
	case <-time.After(time.Second):
		t.Error("progress waited for the projections' lock")
	}
	k.mu.Unlock()
	<-recorded
	// The request has one token left, and one token ahead the next request holds 8 + 1.
	require.Equal(t, int64(1+9), projectedAfterScrapes(k, "a"))
}
