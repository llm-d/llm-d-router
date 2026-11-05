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

// The flow-controller integration test (the CP3 re-review's requirement): requests driven
// through the REAL controller (EnqueueAndWait → the queues → this plugin's Pick), proving
// dispatch, rotation and newcomer entry on the live path — not the unit smoke API.

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"

	contractmocks "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/controller"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	fcTypes "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/types"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/bandselection"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/fcfs"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/threadidentity"
	testutils "github.com/llm-d/llm-d-router/test/utils"
)

// unsaturatedDetector never saturates (the 80% gate is the stack's concern; this test
// exercises the fairness path only).
type unsaturatedDetector struct {
	flowcontrol.SaturationDetector
}

func (d *unsaturatedDetector) Saturation(context.Context, []datalayer.Endpoint) float64 { return 0 }

func (d *unsaturatedDetector) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "test-detector", Name: "unsaturated"}
}

// intRequest is a request carrying an agent-identity InferenceRequest (the session) and a wire
// size (the estimate input).
type intRequest struct {
	id     string
	key    flowcontrol.FlowKey
	size   uint64
	infReq *fwksched.InferenceRequest
}

func (r *intRequest) FlowKey() flowcontrol.FlowKey                 { return r.key }
func (r *intRequest) ByteSize() uint64                             { return r.size }
func (r *intRequest) InitialEffectiveTTL() time.Duration           { return 5 * time.Minute }
func (r *intRequest) ID() string                                   { return r.id }
func (r *intRequest) GetMetadata() map[string]any                  { return nil }
func (r *intRequest) InferencePoolName() string                    { return "test-pool" }
func (r *intRequest) ModelName() string                            { return "test-model" }
func (r *intRequest) TargetModelName() string                      { return "test-target" }
func (r *intRequest) InferenceRequest() *fwksched.InferenceRequest { return r.infReq }
func (r *intRequest) ReceivedTimestamp() time.Time                 { return time.Now() }

// newIntRequest builds a request the way the real stack does: thread identity has already
// derived the THREAD id (the queue key), the token producer has already rendered the tokens
// (the exact chain the scheduler costs), and the client's session rides as an attribute.
// blocks is the prompt's size in 64-token blocks.
func newIntRequest(thread, id string, blocks uint64) *intRequest {
	size := blocks * 256 // the wire size a prompt of this length has at 4 bytes/token
	ir := &fwksched.InferenceRequest{RequestID: id, RequestSizeBytes: int(size), FairnessID: thread}
	ir.PutAttribute(threadidentity.ThreadIdentityKey, thread)
	ir.PutAttribute(agentidentity.AgentIdentityKey, thread)
	ir.PutAttribute(threadidentity.ThreadArrivalKey, threadidentity.ThreadArrival{
		Outcome: threadidentity.OutcomeNew, Turn: 1})
	ir.Body = &fwkrh.InferenceRequestBody{
		ChatCompletions: &fwkrh.ChatCompletionsRequest{},
		TokenizedRequest: &fwkrh.TokenizedRequest{
			Prompts: []fwkrh.PromptTokens{{TokenIDs: tokenRun(int(seedFor(thread)%1000003)+1, int(blocks)*64)}},
		},
	}
	// the blocks thread identity publishes for these tokens: the cost the policy charges
	ir.PutAttribute(threadidentity.ThreadBlocksKey, blocksOfRequest(ir, 64))
	return &intRequest{id: id, key: flowcontrol.FlowKey{ID: thread, Priority: 0}, size: size, infReq: ir}
}

// newIntegrationHarness wires the real controller with the plugin as the band's fairness
// policy (the integration_helpers pattern, minimal).
func newIntegrationHarness(t *testing.T, plugin fwkplugin.Plugin) (*controller.FlowController, *registry.FlowRegistry) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	fp, ok := plugin.(flowcontrol.FairnessPolicy)
	require.True(t, ok, "the plugin must implement FairnessPolicy")

	fcfsPlugin, err := fcfsFactory()
	require.NoError(t, err)
	defaults := registry.PriorityBandPolicyDefaults{
		OrderingPolicy: fcfsPlugin,
		FairnessPolicy: fp,
	}
	band, err := registry.NewPriorityBandConfig(0, defaults, registry.WithBandMaxBytes(1<<40))
	require.NoError(t, err)
	regCfg, err := registry.NewConfig(defaults, registry.WithPriorityBand(band))
	require.NoError(t, err)
	reg := registry.NewFlowRegistry(regCfg, logr.Discard())
	go reg.RunMaintenanceLoop(ctx)

	fc := controller.NewFlowController(ctx, "groupsched-integration-test",
		&controller.Config{
			DefaultRequestTTL:        5 * time.Minute,
			NoEndpointRequestTTL:     5 * time.Minute,
			ExpiryCleanupInterval:    10 * time.Millisecond,
			EnqueueChannelBufferSize: 100,
		},
		controller.Deps{
			Registry:           reg,
			SaturationDetector: &unsaturatedDetector{},
			EndpointCandidates: &contractmocks.MockEndpointCandidates{},
			UsageLimitPolicy:   usagelimits.DefaultPolicy(),
			// The dispatch cycle ranks the bands through the BandSelectionPolicy; the stock strict
			// policy is what an arm that names no band policy loads.
			BandSelectionPolicy: bandselection.DefaultPolicy(),
		})
	t.Cleanup(func() { cancel(); time.Sleep(50 * time.Millisecond) })
	time.Sleep(10 * time.Millisecond)
	return fc, reg
}

// enqueueAndDispatch is EnqueueAndWait plus the dispatch-side hook the director runs the
// moment a request leaves the queue: PreRequest.
func enqueueAndDispatch(t *testing.T, fc *controller.FlowController, gs *GroupScheduler, req *intRequest) (fcTypes.QueueOutcome, error) {
	t.Helper()
	outcome, err := fc.EnqueueAndWait(context.Background(), req)
	if outcome == fcTypes.QueueOutcomeDispatched {
		if perr := gs.PreRequest(context.Background(), req.InferenceRequest(), nil); perr != nil {
			return outcome, perr
		}
	}
	return outcome, err
}

// fcfsFactory builds the FCFS ordering policy (the band default).
func fcfsFactory() (flowcontrol.OrderingPolicy, error) {
	handle := testutils.NewTestHandle(context.Background())
	p, err := fcfs.FCFSOrderingPolicyFactory("fcfs", nil, handle)
	if err != nil {
		return nil, err
	}
	return p.(flowcontrol.OrderingPolicy), nil
}

// dispatchResult mirrors the integration helpers.
type dispatchResult struct {
	id      string
	outcome fcTypes.QueueOutcome
	err     error
}

// TestIntegrationDispatchThroughRealController: two sessions with sizes that only one fits
// (the budget); the thread's request dispatches through the REAL controller; after the thread
// goes quiet, the blocked waiter is rotated in (the engine-idle path) and dispatches too.
func TestIntegrationDispatchThroughRealController(t *testing.T) {
	plugin, err := Factory("gs", nil, nil)
	require.NoError(t, err)
	gs := plugin.(*GroupScheduler)
	// attach the measured capacity (in production: endpoint metrics)
	require.NoError(t, gs.sched.SetNumBlocks(64, 1000)) // budget 950 blocks

	fc, reg := newIntegrationHarness(t, plugin)

	results := make(chan dispatchResult, 4)
	// session "m": 400 blocks; session "w": 600 blocks (400+600 > 950: w cannot co-fit; it waits
	// until m is rotated out) m enqueues and dispatches (a thread; the queue is real).
	go func() {
		req := newIntRequest("m", "m-1", uint64(400))
		outcome, err := enqueueAndDispatch(t, fc, gs, req)
		results <- dispatchResult{id: "m-1", outcome: outcome, err: err}
	}()
	select {
	case r := <-results:
		require.NoError(t, r.err, "m must dispatch")
		require.Equal(t, fcTypes.QueueOutcomeDispatched, r.outcome)
	case <-time.After(5 * time.Second):
		t.Fatal("m never dispatched — the lifecycle deadlock (S0-1) is back")
	}
	require.True(t, gs.sched.Seated("m"), "m must be the seated thread")

	// w enqueues behind the budget (400 + 600 > 950): it stays queued.
	go func() {
		req := newIntRequest("w", "w-1", uint64(600))
		outcome, err := enqueueAndDispatch(t, fc, gs, req)
		results <- dispatchResult{id: "w-1", outcome: outcome, err: err}
	}()
	queuedOrDone := func() bool {
		return reg.Stats().Global.Len >= 1 || len(results) > 0
	}
	require.Eventually(t, queuedOrDone, 2*time.Second, 5*time.Millisecond,
		"w must be queued behind the budget (stats=%+v)", reg.Stats().Global)
	if len(results) > 0 {
		r := <-results
		t.Fatalf("w's enqueue returned early: %+v (budget must hold it)", r)
	}
	// nothing new dispatched yet (w is blocked)
	select {
	case r := <-results:
		t.Fatalf("w dispatched too early (budget violated): %+v", r)
	case <-time.After(300 * time.Millisecond):
	}

	// w is blocked (the budget holds m). m's turn completes and it goes quiet (nothing waiting,
	// nothing in flight): the engine-idle rule rotates w in and it dispatches.
	settle(gs.sched, "m", 400, 0)
	select {
	case r := <-results:
		require.NoError(t, r.err, "w must dispatch after the rotation")
		require.Equal(t, fcTypes.QueueOutcomeDispatched, r.outcome)
	case <-time.After(5 * time.Second):
		t.Fatal("w never dispatched after the rotation")
	}
	require.True(t, gs.sched.Seated("w"), "w must be seated after the rotation")
	fmt.Println("integration: dispatch, budget hold, and rotation all proved through the real controller")
}

// TestIntegrationNewcomerEntry: a newcomer's request dispatches through the real controller
// (the seat + serve path on the live queue).
func TestIntegrationNewcomerEntry(t *testing.T) {
	plugin, err := Factory("gs", nil, nil)
	require.NoError(t, err)
	gs := plugin.(*GroupScheduler)
	require.NoError(t, gs.sched.SetNumBlocks(64, 1000))

	fc, _ := newIntegrationHarness(t, plugin)
	results := make(chan dispatchResult, 1)
	go func() {
		req := newIntRequest("new", "new-1", uint64(100))
		outcome, err := enqueueAndDispatch(t, fc, gs, req)
		results <- dispatchResult{id: "new-1", outcome: outcome, err: err}
	}()
	select {
	case r := <-results:
		require.NoError(t, r.err)
		require.Equal(t, fcTypes.QueueOutcomeDispatched, r.outcome)
	case <-time.After(5 * time.Second):
		t.Fatal("the newcomer never dispatched")
	}
	require.True(t, gs.sched.Seated("new"))
}
