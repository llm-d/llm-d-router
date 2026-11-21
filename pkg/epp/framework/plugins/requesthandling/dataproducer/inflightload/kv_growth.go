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
	"sync"
	"time"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const (
	// The output-length distribution is estimated from the most recent requests: at least
	// kvGrowthMinWindow of them, and kvGrowthRequestsPerRunning for every running request, so that
	// the sample reaches back a few request lifetimes whatever the number of requests in flight.
	kvGrowthMinWindow          = 200
	kvGrowthRequestsPerRunning = 2.5
	// Assumed requests that run to the output cap are added to the sample: at least
	// kvGrowthMinPrior of them, and one for every kvGrowthRequestsPerPrior sampled requests. They
	// are the whole sample until a request completes and keep a share of it afterwards, so the
	// projection still reserves for long outputs after a run of short ones.
	kvGrowthMinPrior         = 20
	kvGrowthRequestsPerPrior = 10
	// kvGrowthMaxSteps bounds the cost of a projection: it looks ahead in at most this many steps
	// of equal length up to the output cap.
	kvGrowthMaxSteps = 256
	// kvGrowthPointsPerStep is how many points of the survival function are kept per step, so
	// that a request is conditioned on the lengths it has outlived within a step.
	kvGrowthPointsPerStep = 16
	// kvGrowthRefreshInterval is how long the survival table is reused. An endpoint's projection
	// is reused for as long as the table, unless a request is dispatched to the endpoint or ends
	// on it.
	kvGrowthRefreshInterval = 100 * time.Millisecond
	// kvGrowthMaxProjection bounds a projection, so that it converts to an integer for any
	// configured cap.
	kvGrowthMaxProjection = 1 << 62
)

// kvGrowthRequest is a request dispatched to an endpoint.
type kvGrowthRequest struct {
	endpoint string
	// prompt is the tokens of the request's prompts, with a copy for each choice. It is reserved
	// in full until the scraped usage includes it: vLLM counts cached blocks as free until a
	// request takes them.
	prompt float64
	// uncached is the part of the prompt the endpoint had not cached. The request is taken to
	// free only that part when it ends, since other requests may share the rest.
	uncached float64
	// sequences is how many sequences the request generates. Its output cap is that of one
	// sequence.
	sequences float64
	outputCap float64
	// output is the tokens streamed so far over all sequences, or the tokens the request ended at.
	output float64
	// startedAt is when the first response chunk arrived, zero before it. The engine has taken
	// the prompt by then.
	startedAt time.Time
	// scrapedAt is the first scrape seen to complete after startedAt, zero before one.
	scrapedAt time.Time
	// sampledAt is the latest scrape seen. seenOutput is output when that scrape and the one seen
	// before it were first seen, and scrapedOutput the part of output the scraped usage includes.
	sampledAt     time.Time
	seenOutput    [2]float64
	scrapedOutput float64
	// observed reports that the request ended at a known full length: the model server completed
	// the response, the request's own output limit did not cut it off, and it generated one
	// sequence.
	observed bool
}

// kvGrowth projects how many more KV cache tokens an endpoint will need at its peak, beyond its
// scraped usage.
//
// A running request holds c_i tokens, its uncached prompt plus the tokens it has generated. Each
// of its m_i sequences keeps growing while it decodes, by at most the r_i tokens left under the
// request's output cap, and the request frees the c_i tokens when it ends. How long it still runs
// is read from the survival function S of recent output lengths (Kaplan-Meier: completed requests
// are observed lengths, running ones and those cut short are known to be longer than what they
// streamed). After t more tokens the expected change in occupancy is
//
//	sum_i [ S_i(t) * (c_i + m_i * min(t, r_i)) - c_i ] + S(t) * (p + m * t)
//
// where S_i(t) = S(output_i + t) / S(output_i), and the last term is the request dispatched next,
// with the mean prompt tokens p and the mean number of sequences m of recent requests. The
// projection is the maximum over t >= 0 of that mean plus z standard deviations, with the
// requests taken as independent. It is bounded by what the requests can still generate under
// their caps, and the prompts and streamed tokens the scraped usage does not include yet are added
// to it.
type kvGrowth struct {
	z               float64
	maxOutputTokens float64
	// The projection looks ahead in steps of stepTokens. The survival function is kept every
	// unitTokens, stride points per step.
	stepTokens, unitTokens float64
	steps, stride          int
	// minWindow and minPrior are kvGrowthMinWindow and kvGrowthMinPrior; tests use smaller samples.
	minWindow, minPrior int

	mu       sync.Mutex
	requests map[string]*kvGrowthRequest            // running, by request ID
	running  map[string]map[string]*kvGrowthRequest // running, by endpoint ID then request ID
	cohort   []*kvGrowthRequest                     // the sampled requests, oldest first
	// projections holds each endpoint's projection until its requests change or the survival
	// table is rebuilt.
	projections map[string]int64

	// survival[i] is the share of sequences longer than i units.
	survival   []float64
	meanPrompt float64
	// meanSequences is the mean number of sequences of the sampled requests.
	meanSequences float64
	tableAt       time.Time

	// Scratch space reused under mu.
	completed, censored []float64
	mean, variance      []float64
}

// kvGrowthRelease ends a request in kvGrowth when its plugin state is evicted.
type kvGrowthRelease struct{ growth *kvGrowth }

func (r kvGrowthRelease) Clone() fwkplugin.StateData { return r }

func (r kvGrowthRelease) OnEvicted(requestID string, _ fwkplugin.StateKey) {
	r.growth.finish(requestID, 0, false)
}

func newKVGrowth(z float64, maxOutputTokens int64) *kvGrowth {
	stepTokens := math.Ceil(float64(maxOutputTokens) / kvGrowthMaxSteps)
	unitTokens := math.Ceil(stepTokens / kvGrowthPointsPerStep)
	stride := int(math.Ceil(stepTokens / unitTokens))
	stepTokens = float64(stride) * unitTokens
	steps := int(math.Ceil(float64(maxOutputTokens) / stepTokens))
	return &kvGrowth{
		z:               z,
		maxOutputTokens: float64(maxOutputTokens),
		stepTokens:      stepTokens,
		unitTokens:      unitTokens,
		steps:           steps,
		stride:          stride,
		minWindow:       kvGrowthMinWindow,
		minPrior:        kvGrowthMinPrior,
		requests:        map[string]*kvGrowthRequest{},
		running:         map[string]map[string]*kvGrowthRequest{},
		projections:     map[string]int64{},
		survival:        make([]float64, steps*stride),
		completed:       make([]float64, steps*stride),
		censored:        make([]float64, steps*stride),
		mean:            make([]float64, steps),
		variance:        make([]float64, steps),
	}
}

// add records a request dispatched to an endpoint. promptTokens is the tokens of its prompts and
// uncachedTokens the part of them the endpoint had not cached. The request generates choices
// sequences for each of its prompts. outputCap is its own output limit for a sequence; zero or
// less means it set none.
func (k *kvGrowth) add(endpointID, requestID string, promptTokens, uncachedTokens, outputCap, prompts, choices int64) {
	k.mu.Lock()
	defer k.mu.Unlock()
	if _, ok := k.requests[requestID]; ok {
		return
	}
	// The choices are not taken to share the blocks of their prompt.
	copies := float64(max(1, choices))
	r := &kvGrowthRequest{
		endpoint: endpointID, prompt: copies * float64(promptTokens), uncached: float64(uncachedTokens),
		sequences: float64(max(1, prompts)) * copies, outputCap: k.maxOutputTokens,
	}
	if outputCap > 0 && float64(outputCap) < k.maxOutputTokens {
		r.outputCap = float64(outputCap)
	}
	k.requests[requestID] = r
	if k.running[endpointID] == nil {
		k.running[endpointID] = map[string]*kvGrowthRequest{}
	}
	k.running[endpointID][requestID] = r
	k.cohort = append(k.cohort, r)
	window := max(k.minWindow, int(kvGrowthRequestsPerRunning*float64(len(k.requests))))
	if len(k.cohort) > window {
		k.cohort = k.cohort[len(k.cohort)-window:]
	}
	delete(k.projections, endpointID)
}

// progress records a response chunk of a running request and the tokens it has generated so far.
func (k *kvGrowth) progress(requestID string, tokens int, now time.Time) {
	k.mu.Lock()
	defer k.mu.Unlock()
	r := k.requests[requestID]
	if r == nil {
		return
	}
	if r.startedAt.IsZero() {
		r.startedAt = now
	}
	r.output = max(r.output, float64(tokens))
}

// promptScraped reports whether the usage scraped at scrapedAt includes the request's prompt. A
// scrape that completes after the first response chunk may have read the usage before the chunk.
// Scrapes of an endpoint run one after another, so the one after it began after the chunk.
func (r *kvGrowthRequest) promptScraped(scrapedAt time.Time) bool {
	if r.startedAt.IsZero() || !scrapedAt.After(r.startedAt) {
		return false
	}
	if r.scrapedAt.IsZero() {
		r.scrapedAt = scrapedAt
	}
	return scrapedAt.After(r.scrapedAt)
}

// outputScraped returns how many of the request's streamed tokens the usage scraped at scrapedAt
// includes. A scrape may have read the usage before the tokens streamed by the time it is first
// seen, and the scrape after the next one began after that.
func (r *kvGrowthRequest) outputScraped(scrapedAt time.Time) float64 {
	if scrapedAt.After(r.sampledAt) {
		r.sampledAt = scrapedAt
		r.scrapedOutput, r.seenOutput[0], r.seenOutput[1] = r.seenOutput[0], r.seenOutput[1], r.output
	}
	return r.scrapedOutput
}

// finish records the end of a request. completionTokens is the reported usage; when it is zero
// the request counts at its streamed tokens. completed reports that the model server ended the
// response on its own; a stream cut short is known only to be longer than what it streamed.
func (k *kvGrowth) finish(requestID string, completionTokens int64, completed bool) {
	k.mu.Lock()
	defer k.mu.Unlock()
	r := k.requests[requestID]
	if r == nil {
		return
	}
	if completionTokens > 0 {
		r.output = float64(completionTokens)
	}
	// A request cut off by its own cap would have run longer under a larger one. The sequences of
	// a request end at different lengths, and only their total is reported.
	r.observed = completed && r.sequences == 1 && (r.output < r.outputCap || r.outputCap >= k.maxOutputTokens)
	if r.output == 0 {
		// A request that ends without output carries no length: an error status, a route whose
		// response is not processed, a stream cut before its first token. It leaves the sample.
		if i := slices.Index(k.cohort, r); i >= 0 {
			k.cohort = slices.Delete(k.cohort, i, i+1)
		}
	}
	delete(k.requests, requestID)
	delete(k.running[r.endpoint], requestID)
	if len(k.running[r.endpoint]) == 0 {
		delete(k.running, r.endpoint)
	}
	// The projection counted on the request freeing its tokens, which the scraped usage shows once
	// it has ended.
	delete(k.projections, r.endpoint)
}

// projected returns the projection for an endpoint in tokens, or zero when no request is running
// on it. scrapedAt is when the endpoint's usage was last scraped.
func (k *kvGrowth) projected(endpointID string, now, scrapedAt time.Time) int64 {
	k.mu.Lock()
	defer k.mu.Unlock()
	running := k.running[endpointID]
	if len(running) == 0 {
		return 0
	}
	k.refreshSurvival(now)
	if tokens, ok := k.projections[endpointID]; ok {
		return tokens
	}

	clear(k.mean)
	clear(k.variance)
	last := len(k.survival) - 1
	// growthLeft is the output the running requests can still generate under their caps.
	// unscraped is the tokens the requests hold that the scraped usage does not include yet.
	var unscraped, growthLeft float64
	for _, r := range running {
		if !r.promptScraped(scrapedAt) {
			unscraped += r.prompt
		}
		unscraped += r.output - r.outputScraped(scrapedAt)
		// The sequences of a request are taken to have generated the same number of tokens.
		output := r.output / r.sequences
		if output >= k.maxOutputTokens {
			// The model server does not stop requests at the configured cap after all: the request
			// is taken to keep its tokens.
			continue
		}
		footprint := r.uncached + r.output
		remaining := max(0, r.outputCap-output)
		growthLeft += r.sequences * remaining
		age := int(output / k.stepTokens)
		capSteps := int(math.Ceil(r.outputCap / k.stepTokens))
		// The request has outlived the lengths up to the first point at or past its output.
		from := int(output / k.unitTokens)
		outlived := k.survival[min(int(math.Ceil(output/k.unitTokens)), last)]
		for g := range k.steps {
			// s is the chance the request is still running g steps from now. A request older than
			// every sampled length is assumed to run to its cap.
			var s float64
			if age+g < capSteps {
				s = 1
				if g > 0 && outlived > 0 {
					s = k.survival[from+g*k.stride] / outlived
				}
			}
			held := footprint + r.sequences*min(float64(g+1)*k.stepTokens, remaining)
			k.mean[g] += s*held - footprint
			k.variance[g] += s * (1 - s) * held * held
		}
	}
	// The next request's prompt is held from the moment it is dispatched, before any running
	// request has had time to end.
	peak := k.meanPrompt
	for g := range k.steps {
		s := k.survival[g*k.stride]
		held := k.meanPrompt + k.meanSequences*float64(g+1)*k.stepTokens
		peak = max(peak, k.mean[g]+s*held+k.z*math.Sqrt(k.variance[g]+s*(1-s)*held*held))
	}
	// The margin can exceed what the requests can still take when few of them are running.
	peak = min(peak, growthLeft+k.meanPrompt+k.meanSequences*k.maxOutputTokens)
	tokens := int64(min(math.Round(unscraped+peak), kvGrowthMaxProjection))
	k.projections[endpointID] = tokens
	return tokens
}

// refreshSurvival recomputes the survival table and the means of the next request from the cohort,
// and drops the projections computed from the previous table.
func (k *kvGrowth) refreshSurvival(now time.Time) {
	if now.Sub(k.tableAt) < kvGrowthRefreshInterval {
		return
	}
	clear(k.completed)
	clear(k.censored)
	var prompts, sequences float64
	for _, r := range k.cohort {
		prompts += r.prompt
		sequences += r.sequences
		// The first point at or past the length of the request's sequences. A length that reaches
		// the last point never leaves the table.
		i := int(math.Ceil(r.output / r.sequences / k.unitTokens))
		switch {
		case i >= len(k.survival):
		case r.observed:
			k.completed[i]++
		default:
			k.censored[i]++
		}
	}
	atRisk := float64(len(k.cohort) + max(k.minPrior, len(k.cohort)/kvGrowthRequestsPerPrior))
	alive := 1.0
	for i := range k.survival {
		// Completions at a point are counted before the requests still running there.
		if k.completed[i] > 0 {
			alive *= 1 - k.completed[i]/atRisk
		}
		atRisk -= k.completed[i] + k.censored[i]
		k.survival[i] = alive
	}
	k.meanPrompt = prompts / float64(len(k.cohort))
	k.meanSequences = sequences / float64(len(k.cohort))
	k.tableAt = now
	clear(k.projections)
}
