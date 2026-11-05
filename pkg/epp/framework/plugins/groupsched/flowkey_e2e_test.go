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

// THE FLOW KEY, END TO END THROUGH THE REAL DIRECTOR.
//
// Thread identity does not write the request's FairnessID field: under the director's own scope
// (#3157) a RequestHeader processor holds a SHALLOW COPY of the request, so an attribute write is
// shared but a struct-field write is dropped. It publishes its thread id on the flow-key attribute
// instead (agentidentity.AgentIdentityKey) — the one attribute the director's derivation reads —
// and these tests drive the REAL Director to prove the queue key that comes out:
//
//	no header                     -> the thread id
//	tenant header, compose ON     -> "tenant/thread"   (skipping the publish would give "tenant")
//	tenant header, compose OFF    -> the tenant as sent (the explicit override)
//	two forks of one prompt       -> two thread ids, two flow keys
//	and the attribution source is "agent_identity": a borrowed label, listed in the README.
//
// Nothing here is a unit argument: the director runs its real header processors under its real
// scope, then its real derivation, and the assertion is on the request it hands to admission.

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	apixv1 "github.com/llm-d/llm-d-router/apix/v1"
	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	"github.com/llm-d/llm-d-router/pkg/common/observability/tracing"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/threadidentity"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"github.com/llm-d/llm-d-router/pkg/epp/requestcontrol"
)

// flowKeyTestModel is the model every request in these tests carries.
const flowKeyTestModel = "test-model"

// flowKeyDatastore is the narrowest Datastore the derivation touches.
type flowKeyDatastore struct{}

func (flowKeyDatastore) PoolGet() (*datalayer.EndpointPool, error) { return nil, nil }

func (flowKeyDatastore) ObjectiveGet(string) *apixv1.InferenceObjective { return nil }

func (flowKeyDatastore) PodList(func(fwkdl.Endpoint) bool) []fwkdl.Endpoint { return nil }

func (flowKeyDatastore) ModelRewriteGet(string) (*v1alpha2.InferenceModelRewriteRule, string) {
	return nil, ""
}

// flowKeyScheduler claims one target and records nothing else.
type flowKeyScheduler struct{ target fwksched.Endpoint }

func (s flowKeyScheduler) Schedule(_ context.Context, _ *fwksched.InferenceRequest,
	_ []fwksched.Endpoint) (*fwksched.SchedulingResult, error) {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{s.target}},
		},
	}, nil
}

// flowKeyAdmission admits everything and captures the flow key it was handed — the value flow
// control would key the queue on.
type flowKeyAdmission struct{ seen *string }

func (a flowKeyAdmission) Admit(_ context.Context, reqCtx *handlers.RequestContext, _ int) error {
	if reqCtx.SchedulingRequest != nil {
		*a.seen = reqCtx.SchedulingRequest.FairnessID
	}
	return nil
}

// flowKeyCandidates offers one endpoint.
type flowKeyCandidates struct{ ep fwkdl.Endpoint }

func (c flowKeyCandidates) Locate(context.Context, map[string]any) []fwkdl.Endpoint {
	return []fwkdl.Endpoint{c.ep}
}

// newFlowKeyDirector builds the real Director around the real thread-identity plugin.
func newFlowKeyDirector(t *testing.T, compose bool) (*requestcontrol.Director, *threadidentity.Plugin, *string) {
	t.Helper()

	// block length 4 so the fixture prompt (8 rendered tokens) is TWO full blocks: with the
	// production length (64) it would be one partial block and no full-block prefix to match,
	// which makes every arrival a NEW thread and the fork case untestable.
	bi := newThreadIdentityAt(t, 4)
	cfg := requestcontrol.NewConfig().
		WithRequestHeaderPlugins(bi).
		// PreRequest is what marks a turn in flight: without it every tip looks free and a
		// second identical prompt would CONTINUE instead of forking.
		WithPreRequestPlugins(bi).
		WithComposeFairnessIDs(compose)

	ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{Name: "pod-a"}, &fwkdl.Metrics{})
	seen := new(string)
	d := requestcontrol.NewDirectorWithConfig(
		flowKeyDatastore{},
		flowKeyScheduler{target: fwksched.NewEndpoint(&fwkdl.EndpointMetadata{Name: "pod-a"}, &fwkdl.Metrics{}, fwkdl.NewAttributes())},
		flowKeyAdmission{seen: seen},
		flowKeyCandidates{ep: ep},
		cfg,
		"test-pool",
	)
	return d, bi, seen
}

// flowKeyRequest builds the handlers.RequestContext the director's HandleRequest takes.
func flowKeyRequest(t *testing.T, id string, headers map[string]string) (*handlers.RequestContext, *fwkrequestBody) {
	t.Helper()
	h := map[string]string{}
	for k, v := range headers {
		h[k] = v
	}
	body := &fwkrequest.InferenceRequestBody{
		ChatCompletions: &fwkrequest.ChatCompletionsRequest{
			Messages: []fwkrequest.Message{{Role: "user", Content: fwkrequest.Content{Raw: `"a prompt for the flow key"`}}},
		},
		Payload: fwkrequest.PayloadMap{"model": flowKeyTestModel, "messages": []any{
			map[string]any{"role": "user", "content": "a prompt for the flow key"},
		}},
	}
	reqCtx := &handlers.RequestContext{
		Request:                  &handlers.Request{ID: id, Headers: h},
		RequestReceivedTimestamp: time.Now(),
		RequestSize:              64,
	}
	return reqCtx, &fwkrequestBody{body: body}
}

// fwkrequestBody is a tiny wrapper so the helper returns one value with a name at the call site.
type fwkrequestBody struct {
	body *fwkrequest.InferenceRequestBody
}

// TestFlowKeyIsTheThreadID: no header, so the thread identity's published thread is the key.
func TestFlowKeyIsTheThreadID(t *testing.T) {
	d, bi, seen := newFlowKeyDirector(t, false)
	reqCtx, b := flowKeyRequest(t, "req-1", nil)

	out, err := d.HandleRequest(context.Background(), reqCtx, b.body)
	require.NoError(t, err)

	id, ok := fwksched.ReadRequestAttribute[string](out.SchedulingRequest, threadidentity.ThreadIdentityKey)
	require.True(t, ok, "thread identity must publish the thread id")
	require.NotEmpty(t, id)
	require.Equal(t, id, out.SchedulingRequest.FairnessID, "the thread id is the flow key")
	require.Equal(t, id, *seen, "the flow key that reaches admission is the thread id")

	// the publish travelled through the director's scope: the shallow copy shares the attribute
	// store, which is the whole reason the thread is published instead of written to the field
	flowKey, ok := fwksched.ReadRequestAttribute[string](out.SchedulingRequest, agentidentity.AgentIdentityKey)
	require.True(t, ok, "the flow-key attribute must be published through the director's scope")
	require.Equal(t, id, flowKey)
	require.NotZero(t, bi.BlockSize(), "the plugin under test is the render-configured thread identity")
}

// TestFlowKeyComposesTenantAndThread: compose ON, a tenant header, and the key is
// "<tenant>/<thread>" — the reason the thread is published even when a header is present.
func TestFlowKeyComposesTenantAndThread(t *testing.T) {
	d, _, seen := newFlowKeyDirector(t, true)
	reqCtx, b := flowKeyRequest(t, "req-2", map[string]string{metadata.FlowFairnessIDKey: "tenant-a"})

	out, err := d.HandleRequest(context.Background(), reqCtx, b.body)
	require.NoError(t, err)

	id, ok := fwksched.ReadRequestAttribute[string](out.SchedulingRequest, threadidentity.ThreadIdentityKey)
	require.True(t, ok)
	require.Equal(t, "tenant-a"+metadata.FairnessIDTenantSeparator+id, out.SchedulingRequest.FairnessID,
		"compose ON must join the tenant and the THREAD — a missing thread would give a bare tenant and one queue per tenant")
	require.Equal(t, out.SchedulingRequest.FairnessID, *seen)
}

// TestFlowKeyKeepsTheTenantUnderAnExplicitHeader: compose OFF, and the tenant is the key as sent.
func TestFlowKeyKeepsTheTenantUnderAnExplicitHeader(t *testing.T) {
	d, _, seen := newFlowKeyDirector(t, false)
	reqCtx, b := flowKeyRequest(t, "req-3", map[string]string{metadata.FlowFairnessIDKey: "tenant-b"})

	out, err := d.HandleRequest(context.Background(), reqCtx, b.body)
	require.NoError(t, err)
	require.Equal(t, "tenant-b", out.SchedulingRequest.FairnessID,
		"an explicit fairness header is the queue key as sent: the operator override")
	require.Equal(t, "tenant-b", *seen)

	// and the thread is STILL published beside it
	id, ok := fwksched.ReadRequestAttribute[string](out.SchedulingRequest, threadidentity.ThreadIdentityKey)
	require.True(t, ok)
	require.NotEmpty(t, id)
}

// TestFlowKeyDiffersForAFork: a fork of the same prompt is a different thread, so a different key.
func TestFlowKeyDiffersForAFork(t *testing.T) {
	d, _, seen := newFlowKeyDirector(t, false)

	first, b1 := flowKeyRequest(t, "req-4", nil)
	out1, err := d.HandleRequest(context.Background(), first, b1.body)
	require.NoError(t, err)
	key1 := out1.SchedulingRequest.FairnessID
	require.NotEmpty(t, key1)

	// the first turn is still IN FLIGHT (PreRequest marked it), so an identical prompt is a
	// fork from the busy tip rather than a continuation
	second, b2 := flowKeyRequest(t, "req-5", nil)
	out2, err := d.HandleRequest(context.Background(), second, b2.body)
	require.NoError(t, err)
	key2 := out2.SchedulingRequest.FairnessID

	require.NotEqual(t, key1, key2, "a fork is a distinct thread, so a distinct flow key")
	require.Equal(t, key2, *seen)

	// and it IS a fork, not an unrelated new thread: the arrival record says so
	arrival, ok := fwksched.ReadRequestAttribute[threadidentity.ThreadArrival](out2.SchedulingRequest, threadidentity.ThreadArrivalKey)
	require.True(t, ok)
	require.Equal(t, threadidentity.OutcomeForked, arrival.Outcome, "the second prompt forks from the busy tip")
}

// TestFlowKeyAttributionSourceIsAgentIdentity pins the README fact: the key is the thread id, but the
// attribution source READS "agent_identity" — the label is borrowed from the attribute's own name.
// The assertion is on what the director actually recorded for this request, not on a constant.
func TestFlowKeyAttributionSourceIsAgentIdentity(t *testing.T) {
	d, _, _ := newFlowKeyDirector(t, false)
	reqCtx, b := flowKeyRequest(t, "req-6", nil)

	ctx := tracing.BeginRequestAttribution(context.Background(), "req-6")
	out, err := d.HandleRequest(ctx, reqCtx, b.body)
	require.NoError(t, err)
	id, ok := fwksched.ReadRequestAttribute[string](out.SchedulingRequest, threadidentity.ThreadIdentityKey)
	require.True(t, ok)

	attributedID, source, ok := tracing.RequestAttribution(ctx)
	require.True(t, ok, "the request began attribution")
	require.Equal(t, id, attributedID, "the attributed flow identity is the thread id")
	require.Equal(t, tracing.AttributionSourceAgentIdentity, source,
		"the source label reads agent_identity: the thread rides upstream's own flow-key attribute")
}
