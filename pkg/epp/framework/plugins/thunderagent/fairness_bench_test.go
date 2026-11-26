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

package thunderagent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/types"

	eppdatalayer "github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/sessionstate"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/requestheader/agentidentity"
	"github.com/llm-d/llm-d-router/test/utils"
)

const (
	benchReqSizeBytes = 4000
	benchTurnTokens   = 1000
)

type benchRig struct {
	ctx      context.Context
	agent    *ThunderAgent
	producer *sessionstate.Producer
	id       *agentidentity.Plugin
	eps      []fwkdl.Endpoint
	n        int
	mu       sync.Mutex
}

func newBenchRig(b *testing.B, pods, sessions, capTokens int, grace time.Duration) *benchRig {
	b.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	b.Cleanup(cancel)

	handle := utils.NewTestHandle(ctx)
	idRaw := json.NewDecoder(strings.NewReader(`{"additionalSessionHeaders": ["x-session-id"]}`))
	idPlg, err := agentidentity.PluginFactory(agentidentity.PluginType, idRaw, handle)
	if err != nil {
		b.Fatal(err)
	}
	handle.AddPlugin(idPlg.TypedName().Name, idPlg)

	raw := json.NewDecoder(strings.NewReader(fmt.Sprintf(
		`{"fallbackCapacityTokens": %d, "idleGracePeriod": %q, "forceAdmitTimeout": "1h"}`,
		capTokens, grace.String(),
	)))
	tp, err := Factory("thunder", raw, handle)
	if err != nil {
		b.Fatal(err)
	}
	agent := tp.(*ThunderAgent)
	handle.AddPlugin(agent.TypedName().Name, agent)

	defaultProducers := map[string]string{
		sessionstate.SessionStateDataKey.String():     sessionstate.SessionStateProducerType,
		sessionstate.EndpointSessionsDataKey.String(): sessionstate.SessionStateProducerType,
	}
	factories := map[string]fwkplugin.FactoryFunc{sessionstate.SessionStateProducerType: sessionstate.Factory}
	if err := eppdatalayer.CreateMissingDataProducers(ctx, defaultProducers, factories, handle); err != nil {
		b.Fatal(err)
	}
	if _, err := eppdatalayer.ValidateAndOrderDataDependencies(handle.GetAllPlugins()); err != nil {
		b.Fatal(err)
	}
	eppdatalayer.RegisterScopeSpecs(handle.GetAllPlugins())

	producer := handle.Plugin(sessionstate.SessionStateProducerType).(*sessionstate.Producer)
	runtime := eppdatalayer.NewRuntime(0)
	if err := producer.RegisterDependencies(runtime); err != nil {
		b.Fatal(err)
	}
	if err := runtime.Configure(nil, logr.Discard()); err != nil {
		b.Fatal(err)
	}

	r := &benchRig{
		ctx:      ctx,
		agent:    agent,
		producer: producer,
		id:       idPlg.(*agentidentity.Plugin),
	}
	for i := range pods {
		ep := runtime.NewEndpoint(ctx, &fwkdl.EndpointMetadata{
			ID: types.NamespacedName{Namespace: "default", Name: fmt.Sprintf("pod-%d", i)},
		})
		r.eps = append(r.eps, ep)
	}
	for i := range sessions {
		req := r.newReq(fmt.Sprintf("s%d", i))
		r.dispatch(req, r.eps[i%pods])
	}
	agent.Saturation(ctx, r.eps)
	return r
}

func (r *benchRig) newReq(session string) *fwksched.InferenceRequest {
	r.mu.Lock()
	r.n++
	n := r.n
	r.mu.Unlock()
	req := &fwksched.InferenceRequest{
		RequestID:        fmt.Sprintf("req-%d", n),
		Headers:          map[string]string{"x-session-id": session},
		RequestSizeBytes: benchReqSizeBytes,
	}
	if err := r.id.RequestHeader(r.ctx, req); err != nil {
		panic(err)
	}
	if err := r.producer.RequestHeader(r.ctx, req); err != nil {
		panic(err)
	}
	return req
}

func (r *benchRig) dispatch(req *fwksched.InferenceRequest, ep fwkdl.Endpoint) {
	target := fwksched.NewEndpointFromSnapshot(ep.GetMetadata(), ep.GetMetrics(), ep.GetAttributes())
	res := &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"default": {TargetEndpoints: []fwksched.Endpoint{target}},
		},
	}
	if err := r.producer.PreRequest(r.ctx, req, res); err != nil {
		panic(err)
	}
	r.producer.ResponseBody(r.ctx, req, &fwkrh.Response{
		EndOfStream:      true,
		TerminationCause: fwkrh.TerminationCauseNatural,
		Usage:            fwkrequest.Usage{TotalTokens: benchTurnTokens},
	}, nil)
}

func (r *benchRig) queueOf(session string) fwkfc.FlowQueueAccessor {
	req := r.newReq(session)
	fcReq := fwkfcmocks.NewMockFlowControlRequest(uint64(req.RequestSizeBytes), req.RequestID, fwkfc.FlowKey{ID: session})
	fcReq.InferenceRequestV = req
	return &fwkfcmocks.MockFlowQueueAccessor{
		LenV:     1,
		FlowKeyV: fwkfc.FlowKey{ID: session},
		PeekV:    &fwkfcmocks.MockQueueItemAccessor{EnqueueTimeV: time.Now(), OriginalRequestV: fcReq},
	}
}

func benchPick(b *testing.B, pods, sessions, queues, paused int, grace time.Duration, held bool, writers int) {
	capTokens := 1_000_000_000
	if held {
		active := max(sessions-paused, 1)
		capTokens = (active / pods) * benchTurnTokens
	}
	r := newBenchRig(b, pods, sessions, capTokens, grace)
	if paused > 0 {
		r.agent.mu.Lock()
		for i := range min(paused, sessions) {
			sid := fmt.Sprintf("s%d", i)
			l := r.agent.ensureLocalLocked(sid)
			l.endpointHint = fmt.Sprintf("default/pod-%d", i%pods)
			l.paused = true
			l.pausedAtTurns = 1
		}
		r.agent.mu.Unlock()
	}
	qs := make([]fwkfc.FlowQueueAccessor, queues)
	for i := range queues {
		if held {
			qs[i] = r.queueOf(fmt.Sprintf("new%d", i))
		} else {
			qs[i] = r.queueOf(fmt.Sprintf("s%d", i))
		}
	}
	band := bandOf(qs...)
	ctx := context.Background()

	stop := make(chan struct{})
	var wg sync.WaitGroup
	var ops int64
	var opsMu sync.Mutex
	for w := range writers {
		wg.Add(1)
		go func() {
			defer wg.Done()
			local := int64(0)
			i := w * 7
			for {
				select {
				case <-stop:
					opsMu.Lock()
					ops += local
					opsMu.Unlock()
					return
				default:
				}
				i++
				idx := 100 + (i % (sessions - 100))
				req := r.newReq(fmt.Sprintf("s%d", idx))
				r.dispatch(req, r.eps[idx%pods])
				local++
			}
		}()
	}

	b.ResetTimer()
	start := time.Now()
	for b.Loop() {
		got, err := r.agent.Pick(ctx, band)
		if err != nil {
			b.Fatal(err)
		}
		if held && got != nil {
			b.Fatal("expected hold")
		}
	}
	elapsed := time.Since(start)
	close(stop)
	wg.Wait()
	if writers > 0 {
		b.ReportMetric(float64(ops)/elapsed.Seconds(), "turns/s")
	}
}

func BenchmarkPickAdmitted_32p_2000s(b *testing.B) {
	benchPick(b, 32, 2000, 64, 0, time.Hour, false, 0)
}
func BenchmarkPickHeld_32p_2000s(b *testing.B) {
	benchPick(b, 32, 2000, 64, 0, time.Hour, true, 0)
}
func BenchmarkPickAdmitted_32p_20000s(b *testing.B) {
	benchPick(b, 32, 20000, 64, 0, time.Hour, false, 0)
}
func BenchmarkPickHeld_32p_20000s(b *testing.B) {
	benchPick(b, 32, 20000, 64, 0, time.Hour, true, 0)
}
func BenchmarkPickHeld_8p_500s(b *testing.B) {
	benchPick(b, 8, 500, 16, 0, time.Hour, true, 0)
}
func BenchmarkPickHeldWriters_32p_2000s(b *testing.B) {
	benchPick(b, 32, 2000, 64, 0, time.Hour, true, 8)
}
func BenchmarkPickHeldWriters_32p_2000s_DefaultGrace(b *testing.B) {
	benchPick(b, 32, 2000, 64, 0, 30*time.Second, true, 8)
}
func BenchmarkPickHeldPaused_32p_20000s_64q(b *testing.B) {
	benchPick(b, 32, 20000, 64, 5000, 30*time.Second, true, 0)
}
func BenchmarkPickHeldPaused_32p_20000s_500q(b *testing.B) {
	benchPick(b, 32, 20000, 500, 5000, 30*time.Second, true, 0)
}
