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
	"bytes"
	"context"
	"encoding/json"
	"slices"
	"testing"

	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"

	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

const (
	prefillAddress = "10.0.0.1"
	decodeAddress  = "10.0.0.2"
	prefillSource  = prefillAddress + ":8000"
	decodeSource   = decodeAddress + ":8000"
	blockSize      = 64
)

// newKVEventsProducer builds a producer in the production mode (output tokens
// excluded) with KV-event prefill completion on. The name isolates the
// completion counter's label series per test.
func newKVEventsProducer(t testing.TB, name string) *InFlightLoadProducer {
	raw, err := json.Marshal(Config{PrefillCompletionFromKVEvents: true})
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	p, err := InFlightLoadProducerFactory(name, json.NewDecoder(bytes.NewReader(raw)), testutils.NewTestHandle(ctx))
	require.NoError(t, err)
	return p.(*InFlightLoadProducer)
}

func newEventEndpoint(name, role, address string) *stubSchedulingEndpoint {
	ep := newStubSchedulingEndpointWithRole(name, role)
	ep.metadata.Address = address
	ep.metadata.Port = "8000"
	return ep
}

func pdResult(prefill, decode fwksched.Endpoint) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			profilePrefill: {TargetEndpoints: []fwksched.Endpoint{prefill}},
			"decode":       {TargetEndpoints: []fwksched.Endpoint{decode}},
		},
	}
}

func seq(start, n int) []uint32 {
	tokens := make([]uint32, n)
	for i := range tokens {
		tokens[i] = uint32(start + i) //nolint:gosec // test values
	}
	return tokens
}

func newStampedRequest(id string, prompts ...[]uint32) *fwksched.InferenceRequest {
	tokenized := &fwkrequest.TokenizedRequest{}
	for _, prompt := range prompts {
		tokenized.Prompts = append(tokenized.Prompts, fwkrequest.PromptTokens{TokenIDs: prompt})
	}
	return &fwksched.InferenceRequest{
		RequestID: id,
		Headers:   map[string]string{},
		Body:      &fwkrequest.InferenceRequestBody{TokenizedRequest: tokenized},
	}
}

func storeBatch(stamp string, tokens []uint32) kvevents.EventBatch {
	return kvevents.EventBatch{Events: []kvevents.GenericEvent{&kvevents.BlockStoredEvent{
		SessionID:   &stamp,
		BlockSize:   blockSize,
		Tokens:      tokens,
		BlockHashes: make([]uint64, len(tokens)/blockSize),
	}}}
}

func completions(name, signal string) float64 {
	return promtestutil.ToFloat64(prefillCompletions.WithLabelValues(name, signal))
}

func pendingWatch(name string) float64 {
	return promtestutil.ToFloat64(prefillPending.WithLabelValues(name))
}

func TestKVEventPrefillCompletion_ReleasesWhenDecodeStoresLastBlock(t *testing.T) {
	t.Parallel()
	const name = "kv-release"
	producer := newKVEventsProducer(t, name)
	ctx := context.Background()
	prefill := newEventEndpoint("prefill-0", bylabel.RolePrefill, prefillAddress)
	decode := newEventEndpoint("decode-0", bylabel.RoleDecode, decodeAddress)
	prefillID, decodeID := fullEndpointName("prefill-0"), fullEndpointName("decode-0")

	prompt := seq(0, 200) // last full block: tokens 128..191
	req := newStampedRequest("req-1", prompt)
	res := pdResult(prefill, decode)
	require.NoError(t, producer.PreRequest(ctx, req, res))
	require.Equal(t, "req-1", req.Headers[sessionIDHeader], "request carries its id as the engine session stamp")
	require.Equal(t, int64(1), producer.requestTracker.get(prefillID))
	require.Equal(t, int64(200), producer.tokenTracker.get(prefillID))
	require.Equal(t, int64(200), producer.tokenTracker.get(decodeID))
	require.Equal(t, float64(1), pendingWatch(name))

	// The prefill engine computes and stores every block: it still holds them for
	// the decode engine's pull, so the prefill entry stays.
	require.NoError(t, producer.ProcessEvents(ctx, kvevents.EventSource{Endpoint: prefillSource}, storeBatch("req-1", prompt[:192])))
	require.Equal(t, int64(1), producer.requestTracker.get(prefillID))
	require.Equal(t, int64(200), producer.tokenTracker.get(prefillID))
	require.Equal(t, float64(0), completions(name, signalKVEvent))

	// The decode engine stores the pulled blocks: the prefill entry is released
	// in full and the decode entry's tokens are freed, as at first byte.
	require.NoError(t, producer.ProcessEvents(ctx, kvevents.EventSource{Endpoint: decodeSource}, storeBatch("req-1", prompt[:192])))
	require.Equal(t, int64(0), producer.requestTracker.get(prefillID))
	require.Equal(t, int64(0), producer.tokenTracker.get(prefillID))
	require.Equal(t, int64(1), producer.requestTracker.get(decodeID))
	require.Equal(t, int64(0), producer.tokenTracker.get(decodeID))
	require.Equal(t, float64(1), completions(name, signalKVEvent))
	require.Equal(t, float64(0), pendingWatch(name))

	// First byte finds the request already released and counts nothing.
	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &fwkrh.Response{StartOfStream: true}, nil)
	require.Equal(t, float64(0), completions(name, signalFirstByte))
	producer.ResponseBody(ctx, req, &fwkrh.Response{EndOfStream: true}, nil)
	require.Equal(t, int64(0), producer.requestTracker.get(decodeID))
}

func TestKVEventPrefillCompletion_IgnoresOtherBlocks(t *testing.T) {
	t.Parallel()
	producer := newKVEventsProducer(t, "kv-other-blocks")
	ctx := context.Background()
	prefill := newEventEndpoint("prefill-0", bylabel.RolePrefill, prefillAddress)
	decode := newEventEndpoint("decode-0", bylabel.RoleDecode, decodeAddress)
	prefillID := fullEndpointName("prefill-0")

	prompt := seq(0, 200)
	req := newStampedRequest("req-2", prompt)
	require.NoError(t, producer.PreRequest(ctx, req, pdResult(prefill, decode)))

	source := kvevents.EventSource{Endpoint: decodeSource}
	// A store of another request under a different stamp is not this request's.
	require.NoError(t, producer.ProcessEvents(ctx, source, storeBatch("someone-else", prompt[128:192])))
	// A store whose trailing block is not the prompt's last full block completes nothing.
	require.NoError(t, producer.ProcessEvents(ctx, source, storeBatch("req-2", prompt[:64])))
	// The last block reported from an engine the request was not dispatched to is ignored.
	require.NoError(t, producer.ProcessEvents(ctx, kvevents.EventSource{Endpoint: "10.0.0.9:8000"}, storeBatch("req-2", prompt[128:192])))
	require.Equal(t, int64(1), producer.requestTracker.get(prefillID))
	require.Equal(t, int64(200), producer.tokenTracker.get(prefillID))
}

func TestKVEventPrefillCompletion_ClientSessionIDIsTheStamp(t *testing.T) {
	t.Parallel()
	producer := newKVEventsProducer(t, "kv-client-stamp")
	ctx := context.Background()
	prefill := newEventEndpoint("prefill-0", bylabel.RolePrefill, prefillAddress)
	decode := newEventEndpoint("decode-0", bylabel.RoleDecode, decodeAddress)
	prefillID := fullEndpointName("prefill-0")

	prompt := seq(0, 200)
	req := newStampedRequest("req-3", prompt)
	req.Body.Payload = fwkrequest.PayloadMap{sessionIDField: "client-session"}
	require.NoError(t, producer.PreRequest(ctx, req, pdResult(prefill, decode)))
	require.Empty(t, req.Headers[sessionIDHeader], "a client session id is left in place")

	source := kvevents.EventSource{Endpoint: decodeSource}
	require.NoError(t, producer.ProcessEvents(ctx, source, storeBatch("req-3", prompt[128:192])))
	require.Equal(t, int64(1), producer.requestTracker.get(prefillID), "the engine echoes the client's id, not the request id")
	require.NoError(t, producer.ProcessEvents(ctx, source, storeBatch("client-session", prompt[128:192])))
	require.Equal(t, int64(0), producer.requestTracker.get(prefillID))
}

func TestSessionStampPrecedence(t *testing.T) {
	t.Parallel()
	header := newStampedRequest("req-h", seq(0, 64))
	header.Headers["X-Session-ID"] = "from-header"
	header.Body.Payload = fwkrequest.PayloadMap{extraArgsField: map[string]any{sessionIDField: "from-xargs"}}
	stamp, own, ok := sessionStamp(header)
	require.True(t, ok)
	require.False(t, own)
	require.Equal(t, "from-header", stamp)

	xargs := newStampedRequest("req-x", seq(0, 64))
	xargs.Body.Payload = fwkrequest.PayloadMap{extraArgsField: map[string]any{sessionIDField: "from-xargs"}}
	stamp, own, ok = sessionStamp(xargs)
	require.True(t, ok)
	require.False(t, own)
	require.Equal(t, "from-xargs", stamp)
	require.Empty(t, xargs.Headers[sessionIDHeader])

	fresh := newStampedRequest("req-f", seq(0, 64))
	stamp, own, ok = sessionStamp(fresh)
	require.True(t, ok)
	require.True(t, own, "the producer's own stamp is the request id")
	require.Equal(t, "req-f", stamp)
	require.Equal(t, "req-f", fresh.Headers[sessionIDHeader])

	none := newStampedRequest("req-n", seq(0, 64))
	none.Headers = nil
	_, _, ok = sessionStamp(none)
	require.False(t, ok, "no header map to stamp")
}

func TestKVEventPrefillCompletion_SharedClientSessionKeepsEveryRequest(t *testing.T) {
	t.Parallel()
	producer := newKVEventsProducer(t, "kv-shared-session")
	ctx := context.Background()
	prefill := newEventEndpoint("prefill-0", bylabel.RolePrefill, prefillAddress)
	decode := newEventEndpoint("decode-0", bylabel.RoleDecode, decodeAddress)
	prefillID := fullEndpointName("prefill-0")

	promptA, promptB := seq(0, 200), seq(1000, 200)
	for id, prompt := range map[string][]uint32{"req-a": promptA, "req-b": promptB} {
		req := newStampedRequest(id, prompt)
		req.Body.Payload = fwkrequest.PayloadMap{sessionIDField: "shared"}
		require.NoError(t, producer.PreRequest(ctx, req, pdResult(prefill, decode)))
	}
	require.Equal(t, int64(2), producer.requestTracker.get(prefillID))
	require.Equal(t, float64(2), pendingWatch("kv-shared-session"))

	source := kvevents.EventSource{Endpoint: decodeSource}
	require.NoError(t, producer.ProcessEvents(ctx, source, storeBatch("shared", promptB[128:192])))
	require.Equal(t, int64(1), producer.requestTracker.get(prefillID), "B's last block completes B only")
	require.Equal(t, float64(1), pendingWatch("kv-shared-session"))
	require.NoError(t, producer.ProcessEvents(ctx, source, storeBatch("shared", seq(5000, 64))))
	require.Equal(t, int64(1), producer.requestTracker.get(prefillID), "a block outside the prompt under a shared stamp belongs to another request")
	require.NoError(t, producer.ProcessEvents(ctx, source, storeBatch("shared", promptA[128:192])))
	require.Equal(t, int64(0), producer.requestTracker.get(prefillID))
	require.Equal(t, float64(0), pendingWatch("kv-shared-session"))
}

func TestKVEventPrefillCompletion_GeneratedBlockCompletesHeldPrompt(t *testing.T) {
	t.Parallel()
	producer := newKVEventsProducer(t, "kv-generated-block")
	ctx := context.Background()
	prefill := newEventEndpoint("prefill-0", bylabel.RolePrefill, prefillAddress)
	decode := newEventEndpoint("decode-0", bylabel.RoleDecode, decodeAddress)
	prefillID := fullEndpointName("prefill-0")

	prompt := seq(0, 200)
	req := newStampedRequest("req-g", prompt)
	require.NoError(t, producer.PreRequest(ctx, req, pdResult(prefill, decode)))
	source := kvevents.EventSource{Endpoint: decodeSource}
	require.NoError(t, producer.ProcessEvents(ctx, source, storeBatch("req-g", prompt[64:128])))
	require.Equal(t, int64(1), producer.requestTracker.get(prefillID), "a prompt block other than the last leaves the request pending")

	generated := append(slices.Clone(prompt[192:]), seq(9000, 56)...)
	require.NoError(t, producer.ProcessEvents(ctx, source, storeBatch("req-g", generated)))
	require.Equal(t, int64(0), producer.requestTracker.get(prefillID), "the block after the prompt completes a prompt the engine already held")
	require.Equal(t, float64(1), completions("kv-generated-block", signalKVEvent))
	require.Equal(t, float64(0), pendingWatch("kv-generated-block"))
}

func TestKVEventPrefillCompletion_ResetFallsBackToFirstByte(t *testing.T) {
	t.Parallel()
	const name = "kv-reset"
	producer := newKVEventsProducer(t, name)
	ctx := context.Background()
	prefill := newEventEndpoint("prefill-0", bylabel.RolePrefill, prefillAddress)
	decode := newEventEndpoint("decode-0", bylabel.RoleDecode, decodeAddress)
	prefillID, decodeID := fullEndpointName("prefill-0"), fullEndpointName("decode-0")

	prompt := seq(0, 200)
	req := newStampedRequest("req-4", prompt)
	res := pdResult(prefill, decode)
	require.NoError(t, producer.PreRequest(ctx, req, res))

	source := kvevents.EventSource{Endpoint: decodeSource}
	require.NoError(t, producer.Reset(ctx, source))
	require.NoError(t, producer.ProcessEvents(ctx, source, storeBatch("req-4", prompt[:192])))
	require.Equal(t, int64(1), producer.requestTracker.get(prefillID), "a reset endpoint's events no longer release")
	require.Equal(t, int64(200), producer.tokenTracker.get(prefillID))

	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &fwkrh.Response{StartOfStream: true}, nil)
	require.Equal(t, int64(0), producer.requestTracker.get(prefillID))
	require.Equal(t, int64(0), producer.tokenTracker.get(decodeID))
	require.Equal(t, float64(1), completions(name, signalFirstByte))
	require.Equal(t, float64(0), completions(name, signalKVEvent))
}

func TestKVEventPrefillCompletion_EndOfStreamDropsWatch(t *testing.T) {
	t.Parallel()
	const name = "kv-eos"
	producer := newKVEventsProducer(t, name)
	ctx := context.Background()
	prefill := newEventEndpoint("prefill-0", bylabel.RolePrefill, prefillAddress)
	decode := newEventEndpoint("decode-0", bylabel.RoleDecode, decodeAddress)

	req := newStampedRequest("req-5", seq(0, 200))
	res := pdResult(prefill, decode)
	require.NoError(t, producer.PreRequest(ctx, req, res))
	require.NotNil(t, producer.watch.lookup(decodeSource, "req-5"))
	require.Nil(t, producer.watch.lookup(prefillSource, "req-5"), "only the decode engine's events are watched")
	require.Equal(t, float64(1), pendingWatch(name))

	// A request that ends without either signal (an aborted or failed response)
	// leaves no watch behind: the entries' eviction drops it.
	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &fwkrh.Response{EndOfStream: true}, nil)
	require.Nil(t, producer.watch.lookup(decodeSource, "req-5"))
	require.Empty(t, producer.watch.byRequest)
	require.Empty(t, producer.watch.byEndpoint)
	require.Equal(t, float64(0), pendingWatch(name))
	require.Equal(t, float64(0), completions(name, signalKVEvent))
	require.Equal(t, float64(0), completions(name, signalFirstByte))
}

func TestKVEventPrefillCompletion_MultiPrompt(t *testing.T) {
	t.Parallel()
	producer := newKVEventsProducer(t, "kv-multi")
	ctx := context.Background()
	prefill := newEventEndpoint("prefill-0", bylabel.RolePrefill, prefillAddress)
	decode := newEventEndpoint("decode-0", bylabel.RoleDecode, decodeAddress)
	prefillID := fullEndpointName("prefill-0")

	first, second, short := seq(0, 200), seq(1000, 100), seq(5000, 30)
	req := newStampedRequest("req-6", first, second, short)
	require.NoError(t, producer.PreRequest(ctx, req, pdResult(prefill, decode)))

	source := kvevents.EventSource{Endpoint: decodeSource}
	require.NoError(t, producer.ProcessEvents(ctx, source, storeBatch("req-6", first[:192])))
	require.Equal(t, int64(1), producer.requestTracker.get(prefillID), "second prompt not stored yet")
	// The 30-token prompt never fills a block; the second prompt's only full block completes the request.
	require.NoError(t, producer.ProcessEvents(ctx, source, storeBatch("req-6", second[:64])))
	require.Equal(t, int64(0), producer.requestTracker.get(prefillID))
}

func TestKVEventPrefillCompletion_ShortPromptCompletesAtFirstByte(t *testing.T) {
	t.Parallel()
	const name = "kv-short"
	producer := newKVEventsProducer(t, name)
	ctx := context.Background()
	prefill := newEventEndpoint("prefill-0", bylabel.RolePrefill, prefillAddress)
	decode := newEventEndpoint("decode-0", bylabel.RoleDecode, decodeAddress)
	prefillID := fullEndpointName("prefill-0")

	// 13 prompt tokens: the first block the engine stores is mostly generated tokens.
	prompt := seq(0, 13)
	req := newStampedRequest("req-8", prompt)
	res := pdResult(prefill, decode)
	require.NoError(t, producer.PreRequest(ctx, req, res))
	generated := append(append([]uint32{}, prompt...), seq(9000, 51)...)
	require.NoError(t, producer.ProcessEvents(ctx, kvevents.EventSource{Endpoint: decodeSource}, storeBatch("req-8", generated)))
	require.Equal(t, int64(1), producer.requestTracker.get(prefillID), "a generated block completes nothing")
	require.Equal(t, float64(0), completions(name, signalKVEvent))

	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &fwkrh.Response{StartOfStream: true}, nil)
	require.Equal(t, int64(0), producer.requestTracker.get(prefillID))
	require.Equal(t, float64(1), completions(name, signalFirstByte))
}

func TestKVEventPrefillCompletion_SingleEngineCompletesOnItsOwnEvents(t *testing.T) {
	t.Parallel()
	const name = "kv-single"
	producer := newKVEventsProducer(t, name)
	ctx := context.Background()
	engine := newEventEndpoint("engine-0", bylabel.RolePrefillDecode, prefillAddress)
	engineID := fullEndpointName("engine-0")

	prompt := seq(0, 200)
	req := newStampedRequest("req-9", prompt)
	res := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults:     map[string]*fwksched.ProfileRunResult{"default": {TargetEndpoints: []fwksched.Endpoint{engine}}},
	}
	require.NoError(t, producer.PreRequest(ctx, req, res))
	require.Equal(t, int64(200), producer.tokenTracker.get(engineID))

	// With no transfer, the computing engine's own store of the last block ends
	// the prompt's footprint: its tokens are freed, the request stays until the end.
	require.NoError(t, producer.ProcessEvents(ctx, kvevents.EventSource{Endpoint: prefillSource}, storeBatch("req-9", prompt[:192])))
	require.Equal(t, int64(0), producer.tokenTracker.get(engineID))
	require.Equal(t, int64(1), producer.requestTracker.get(engineID))
	require.Equal(t, float64(1), completions(name, signalKVEvent))
}

func TestKVEventPrefillCompletion_OffByDefault(t *testing.T) {
	t.Parallel()
	producer := newTestProducer(t)
	producer.addEstimatedOutputTokens = false
	ctx := context.Background()
	prefill := newEventEndpoint("prefill-0", bylabel.RolePrefill, prefillAddress)
	decode := newEventEndpoint("decode-0", bylabel.RoleDecode, decodeAddress)
	prefillID := fullEndpointName("prefill-0")

	prompt := seq(0, 200)
	req := newStampedRequest("req-7", prompt)
	res := pdResult(prefill, decode)
	require.NoError(t, producer.PreRequest(ctx, req, res))
	require.Empty(t, req.Headers, "no stamp without the option")
	require.NoError(t, producer.ProcessEvents(ctx, kvevents.EventSource{Endpoint: decodeSource}, storeBatch("req-7", prompt[:192])))
	require.Equal(t, int64(1), producer.requestTracker.get(prefillID))
	require.Equal(t, int64(200), producer.tokenTracker.get(prefillID))

	req.SchedulingResult = res
	producer.ResponseBody(ctx, req, &fwkrh.Response{StartOfStream: true}, nil)
	require.Equal(t, int64(0), producer.requestTracker.get(prefillID))
}
