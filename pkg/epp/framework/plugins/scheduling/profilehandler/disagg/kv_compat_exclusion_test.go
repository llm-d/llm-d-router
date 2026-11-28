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

package disagg

import (
	"context"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/bylabel"
	"github.com/llm-d/llm-d-router/test/utils"
)

type exclusionFixture struct {
	t       *testing.T
	d       *KVCompatPDDecider
	f       *KVCompatFilter
	clock   time.Time
	decode  scheduling.Endpoint
	prefill scheduling.Endpoint
	seq     int
}

func newExclusionFixture(t *testing.T, params KVCompatPDDeciderParameters, decodeKey, prefillKey string) *exclusionFixture {
	t.Helper()
	fx := &exclusionFixture{
		t:       t,
		clock:   time.Unix(1_000_000, 0),
		decode:  kvEndpoint("d", bylabel.RoleDecode, decodeKey, nil),
		prefill: kvEndpoint("p", bylabel.RolePrefill, prefillKey, nil),
	}
	fx.d = newKVCompatDecider(t, &fakeDecider{result: true}, params, fx.decode, fx.prefill)
	fx.d.now = func() time.Time { return fx.clock }
	fx.d.typedName.Name = t.Name()
	fx.d.spec = fx.d.buildSpec()
	fx.f = &KVCompatFilter{typedName: plugin.TypedName{Type: KVCompatFilterPluginType, Name: "f"}, decider: fx.d}
	return fx
}

func (fx *exclusionFixture) schedule() *scheduling.InferenceRequest {
	fx.seq++
	req := makeRequestWithTokens(2000)
	req.RequestID = fx.t.Name() + "-" + string(rune('a'+fx.seq))
	result := &scheduling.SchedulingResult{
		PrimaryProfileName: "decode",
		ProfileResults: map[string]*scheduling.ProfileRunResult{
			"decode":  {TargetEndpoints: []scheduling.Endpoint{fx.decode}},
			"prefill": {TargetEndpoints: []scheduling.Endpoint{fx.prefill}},
		},
	}
	require.NoError(fx.t, fx.d.PreRequest(context.Background(), req, result))
	return req
}

func (fx *exclusionFixture) respond(status string) {
	req := fx.schedule()
	fx.d.ResponseHeader(context.Background(), req, &fwkrh.Response{Headers: map[string]string{reqcommon.PseudoHeaderStatus: status}}, nil)
	fx.d.ResponseBody(context.Background(), req, &fwkrh.Response{EndOfStream: true, TerminationCause: fwkrh.TerminationCauseNatural,
		Usage: fwkrequest.Usage{PromptTokens: 2000, CompletionTokens: 10}}, nil)
}

func (fx *exclusionFixture) splits() bool {
	return fx.d.disaggregate(context.Background(), makeRequestWithTokens(2000), fx.decode)
}

func (fx *exclusionFixture) filtered() []scheduling.Endpoint {
	req := makeRequestWithTokens(2000)
	req.PutAttribute(PeerEndpointAttributeKey, fx.decode)
	return fx.f.Filter(context.Background(), req, []scheduling.Endpoint{fx.prefill})
}

func TestKVCompatExclusionAfterConsecutiveFailures(t *testing.T) {
	fx := newExclusionFixture(t, KVCompatPDDeciderParameters{}, "k1", "k1")
	before := testutil.ToFloat64(LlmdKVCompatExclusionCount.WithLabelValues(fx.d.typedName.Name))

	fx.respond("503")
	fx.respond("503")
	fx.respond("200")
	fx.respond("503")
	fx.respond("503")
	require.True(t, fx.splits(), "a success resets the count")

	fx.respond("503")
	assert.False(t, fx.splits(), "the third consecutive failure excludes the pair")
	assert.Empty(t, fx.filtered())
	assert.Equal(t, before+1, testutil.ToFloat64(LlmdKVCompatExclusionCount.WithLabelValues(fx.d.typedName.Name)))

	fx.clock = fx.clock.Add(time.Minute)
	fx.respond("503")
	fx.respond("503")
	fx.respond("503")
	fx.clock = fx.clock.Add(defaultKVCompatExclusionDuration - time.Minute + time.Second)
	assert.True(t, fx.splits(), "failures during an exclusion do not extend it")
	assert.Len(t, fx.filtered(), 1)
	fx.respond("200")
	assert.Empty(t, fx.d.state.pairs, "expired state is dropped on the next outcome")
}

func TestKVCompatExclusionSignals(t *testing.T) {
	end := func(cause fwkrh.TerminationCause, usage fwkrequest.Usage) *fwkrh.Response {
		return &fwkrh.Response{EndOfStream: true, TerminationCause: cause, Usage: usage}
	}
	tests := []struct {
		name     string
		response *fwkrh.Response
		header   bool
		failure  bool
	}{
		{"error status", &fwkrh.Response{Headers: map[string]string{reqcommon.HeaderStatus: "502"}}, true, true},
		{"status not in failureCodes", &fwkrh.Response{Headers: map[string]string{reqcommon.PseudoHeaderStatus: "429"}}, true, false},
		{"unparsable status", &fwkrh.Response{Headers: map[string]string{reqcommon.PseudoHeaderStatus: "x"}}, true, false},
		{"stream ended with an error", end(fwkrh.TerminationCauseError, fwkrequest.Usage{}), false, true},
		{"prompt read but no token generated", end(fwkrh.TerminationCauseNatural, fwkrequest.Usage{PromptTokens: 9}), false, true},
		{"tokens generated", end("", fwkrequest.Usage{PromptTokens: 9, CompletionTokens: 1}), false, false},
		{"client disconnect", end(fwkrh.TerminationCauseClientDisconnect, fwkrequest.Usage{}), false, false},
		{"not the end of the stream", &fwkrh.Response{TerminationCause: fwkrh.TerminationCauseError}, false, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			fx := newExclusionFixture(t, KVCompatPDDeciderParameters{FailureThreshold: ptr.To(1)}, "k1", "k1")
			req := fx.schedule()
			if tt.header {
				fx.d.ResponseHeader(context.Background(), req, tt.response, nil)
			} else {
				fx.d.ResponseBody(context.Background(), req, tt.response, nil)
			}
			assert.Equal(t, tt.failure, !fx.splits())
		})
	}
}

func TestKVCompatExclusionIgnoresOtherRequests(t *testing.T) {
	fx := newExclusionFixture(t, KVCompatPDDeciderParameters{FailureThreshold: ptr.To(1)}, "k1", "k1")
	ctx := context.Background()
	req := makeRequestWithTokens(2000)
	decodeOnly := &scheduling.SchedulingResult{PrimaryProfileName: "decode", ProfileResults: map[string]*scheduling.ProfileRunResult{
		"decode": {TargetEndpoints: []scheduling.Endpoint{fx.decode}},
	}}
	require.NoError(t, fx.d.PreRequest(ctx, req, decodeOnly))
	require.NoError(t, fx.d.PreRequest(ctx, nil, decodeOnly))
	require.NoError(t, fx.d.PreRequest(ctx, req, nil))
	require.NoError(t, fx.d.PreRequest(ctx, req, &scheduling.SchedulingResult{ProfileResults: map[string]*scheduling.ProfileRunResult{
		"decode": {TargetEndpoints: []scheduling.Endpoint{fx.decode}}, "prefill": {},
	}, PrimaryProfileName: "decode"}))
	fx.d.ResponseHeader(ctx, req, &fwkrh.Response{Headers: map[string]string{reqcommon.PseudoHeaderStatus: "503"}}, nil)
	fx.d.ResponseHeader(ctx, req, nil, nil)
	fx.d.ResponseBody(ctx, req, nil, nil)
	assert.True(t, fx.splits(), "decode-only requests never count toward a pair")

	disabled := newExclusionFixture(t, KVCompatPDDeciderParameters{FailureThreshold: ptr.To(0)}, "k1", "k1")
	disabled.respond("503")
	assert.True(t, disabled.splits())
	assert.True(t, disabled.d.spec.SyncDisabled)
	assert.Same(t, noKVCompatExclusions, disabled.d.exclusionsFor(ctx, disabled.decode.GetMetadata().ID))
}

func TestKVCompatExclusionWithoutKeys(t *testing.T) {
	fx := newExclusionFixture(t, KVCompatPDDeciderParameters{FailureThreshold: ptr.To(1)}, "", "")
	fx.respond("503")
	assert.False(t, fx.splits(), "without keys the prefill endpoint itself is excluded")
	assert.Empty(t, fx.filtered())

	other := kvEndpoint("p2", bylabel.RolePrefill, "", nil)
	require.NoError(t, fx.d.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: dlEndpoint(other)}))
	assert.True(t, fx.splits(), "another prefill endpoint is still usable")

	require.NoError(t, fx.d.Extract(context.Background(), fwkdl.EndpointEvent{Type: fwkdl.EventDelete, Endpoint: dlEndpoint(fx.decode)}))
	assert.Empty(t, fx.d.state.pairs, "deleting the decode endpoint drops its state")
}

// peerSyncer stands in for a cross-replica syncer holding one peer replica's exclusions.
type peerSyncer struct {
	peer *KVCompatExclusions
}

func (s *peerSyncer) TypedName() plugin.TypedName {
	return plugin.TypedName{Type: "peer-syncer", Name: "peer"}
}
func (s *peerSyncer) Set(context.Context, fwkdl.CrossReplicaSpec, string) error { return nil }
func (s *peerSyncer) Get(_ context.Context, spec fwkdl.CrossReplicaSpec, id string) (any, bool, error) {
	return spec.Aggregate([]any{spec.Read(id), s.peer}), true, nil
}
func (s *peerSyncer) Delete(context.Context, fwkdl.StateKey, string) error { return nil }
func (s *peerSyncer) GetOrSet(_ context.Context, _ fwkdl.StateKey, _ string, candidate any) (any, bool, error) {
	return candidate, false, nil
}

func TestKVCompatExclusionAcrossReplicas(t *testing.T) {
	fx := newExclusionFixture(t, KVCompatPDDeciderParameters{}, "k1", "k1")
	handle := utils.NewTestHandle(context.Background())
	syncer := &peerSyncer{peer: &KVCompatExclusions{Until: map[string]int64{"k1": fx.clock.Add(time.Minute).UnixMilli()}}}
	handle.SetCrossReplicaSyncer(syncer)
	fx.d.handle = handle

	assert.False(t, fx.splits(), "an exclusion published by a peer replica applies here")
	assert.Empty(t, fx.filtered())

	fx.clock = fx.clock.Add(2 * time.Minute)
	assert.True(t, fx.splits(), "the peer's exclusion ends at its expiry")

	merged := fx.d.spec.Aggregate([]any{
		&KVCompatExclusions{Until: map[string]int64{"a": 5, "b": 1}},
		&KVCompatExclusions{Until: map[string]int64{"a": 3, "b": 7}},
		nil, "ignored",
	}).(*KVCompatExclusions)
	assert.Equal(t, map[string]int64{"a": 5, "b": 7}, merged.Until)
	one := &KVCompatExclusions{Until: map[string]int64{"a": 5}}
	assert.Same(t, one, fx.d.spec.Aggregate([]any{noKVCompatExclusions, one, nil}), "a single non-empty value is returned as is")
	assert.Same(t, noKVCompatExclusions, fx.d.spec.Aggregate([]any{&KVCompatExclusions{}, nil}))
	ctx, id := context.Background(), fx.decode.GetMetadata().ID
	idStr := id.String()
	syncerOnly := testing.AllocsPerRun(100, func() { _, _, _ = syncer.Get(ctx, fx.d.spec, idStr) })
	assert.Equal(t, syncerOnly, testing.AllocsPerRun(100, func() { fx.d.exclusionsFor(ctx, id) }),
		"reading a peer's exclusions allocates nothing beyond the syncer's own lookup")
	assert.Equal(t, fx.d.spec.StateKey, fx.d.CrossReplicaState().StateKey)

	clone := merged.Clone().(*KVCompatExclusions)
	clone.Until["a"] = 0
	assert.Equal(t, int64(5), merged.Until["a"], "Clone copies the map")
	assert.Nil(t, (*KVCompatExclusions)(nil).Clone())
	assert.NotNil(t, (&kvCompatRequestPair{pair: "x"}).Clone())

	handle.SetCrossReplicaSyncer(nil)
	assert.Nil(t, fx.d.crossReplicaSyncer())
}

func TestKVCompatExclusionConfig(t *testing.T) {
	cfg, err := newKVCompatExclusionConfig(KVCompatPDDeciderParameters{})
	require.NoError(t, err)
	assert.Equal(t, defaultKVCompatFailureThreshold, cfg.threshold)
	assert.Equal(t, defaultKVCompatExclusionDuration, cfg.duration)
	assert.Equal(t, DefaultPrefillProfile, cfg.prefillProfile)
	assert.Len(t, cfg.failureCodes, len(defaultKVCompatFailureCodes))

	cfg, err = newKVCompatExclusionConfig(KVCompatPDDeciderParameters{
		FailureThreshold: ptr.To(2), ExclusionDuration: &metav1.Duration{Duration: time.Second},
		FailureCodes: []int{504}, PrefillProfile: "p"})
	require.NoError(t, err)
	assert.Equal(t, kvCompatExclusionConfig{threshold: 2, duration: time.Second, failureCodes: map[int]struct{}{504: {}}, prefillProfile: "p"}, cfg)

	for name, p := range map[string]KVCompatPDDeciderParameters{
		"negative threshold": {FailureThreshold: ptr.To(-1)},
		"zero duration":      {ExclusionDuration: &metav1.Duration{}},
		"not an error code":  {FailureCodes: []int{200}},
	} {
		_, err := newKVCompatExclusionConfig(p)
		assert.Error(t, err, name)
		_, err = NewKVCompatPDDecider(&fakeDecider{}, p)
		assert.Error(t, err, name)
	}
}
