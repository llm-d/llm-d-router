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

package peerdemand

import (
	"context"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	testclock "k8s.io/utils/clock/testing"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
)

type fakePeers struct {
	queued   map[int]uint64
	stale    bool
	interval time.Duration
}

func (f *fakePeers) QueuedRequests(priority int) (uint64, bool) {
	if f.stale {
		return 0, false
	}
	return f.queued[priority], true
}

func (f *fakePeers) SyncInterval() time.Duration {
	return f.interval
}

// ladder is a delegate with distinct ceilings per band, counting its calls.
type ladder struct {
	calls int
}

func (l *ladder) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "ladder", Name: "ladder"}
}

func (l *ladder) ComputeLimit(_ context.Context, _ float64, _ []int, ceilings []float64) {
	l.calls++
	for i := range ceilings {
		ceilings[i] = 1.0 - 0.1*float64(i)
	}
}

var priorities = []int{80, 60, 40, 0}

func compute(p *Policy) []float64 {
	ceilings := []float64{1, 1, 1, 1}
	p.ComputeLimit(context.Background(), 0.5, priorities, ceilings)
	return ceilings
}

func TestComputeLimit(t *testing.T) {
	t.Parallel()
	tests := []struct {
		name  string
		peers *fakePeers
		want  []float64
	}{
		{name: "no peer view", want: []float64{1.0, 0.9, 0.8, 0.7}},
		{name: "peers idle", peers: &fakePeers{}, want: []float64{1.0, 0.9, 0.8, 0.7}},
		{name: "stale view", peers: &fakePeers{queued: map[int]uint64{80: 4}, stale: true},
			want: []float64{1.0, 0.9, 0.8, 0.7}},
		{name: "highest band queued on a peer", peers: &fakePeers{queued: map[int]uint64{80: 4}},
			want: []float64{1.0, 0, 0, 0}},
		{name: "middle band queued on a peer", peers: &fakePeers{queued: map[int]uint64{40: 1, 0: 9}},
			want: []float64{1.0, 0.9, 0.8, 0}},
		{name: "lowest band queued on a peer", peers: &fakePeers{queued: map[int]uint64{0: 9}},
			want: []float64{1.0, 0.9, 0.8, 0.7}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			delegate := &ladder{}
			p := New("", delegate)
			if tc.peers != nil {
				p.SetPeerDemand(tc.peers)
			}
			assert.Equal(t, tc.want, compute(p))
			assert.Equal(t, 1, delegate.calls, "the delegate runs once per dispatch cycle")
		})
	}
}

func TestComputeLimitWithoutDelegate(t *testing.T) {
	t.Parallel()
	p := New("", nil)
	p.SetPeerDemand(&fakePeers{queued: map[int]uint64{60: 2}})

	assert.Equal(t, []float64{1, 1, 0, 0}, compute(p))
}

func TestSetPeerDemandNilClearsTheView(t *testing.T) {
	t.Parallel()
	p := New("", nil)
	p.SetPeerDemand(&fakePeers{queued: map[int]uint64{80: 2}})
	p.SetPeerDemand(nil)

	assert.Equal(t, []float64{1, 1, 1, 1}, compute(p))
}

func TestComputeLimitHoldsAfterPeerDemandClears(t *testing.T) {
	t.Parallel()
	start := time.Unix(1000, 0)
	tests := []struct {
		name         string
		releaseDelay *time.Duration
		heldFor      time.Duration
	}{
		{name: "default is two sync intervals", heldFor: 40 * time.Millisecond},
		{name: "configured delay", releaseDelay: ptr(150 * time.Millisecond), heldFor: 150 * time.Millisecond},
		{name: "zero releases at once", releaseDelay: ptr(time.Duration(0))},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			clk := testclock.NewFakePassiveClock(start)
			peers := &fakePeers{queued: map[int]uint64{60: 3}, interval: 20 * time.Millisecond}
			p := New("", &ladder{})
			p.clock = clk
			p.releaseDelay = tc.releaseDelay
			p.SetPeerDemand(peers)
			require.Equal(t, []float64{1.0, 0.9, 0, 0}, compute(p))

			peers.queued = map[int]uint64{}
			if tc.heldFor > 0 {
				clk.SetTime(start.Add(tc.heldFor - time.Millisecond))
				assert.Equal(t, []float64{1.0, 0.9, 0, 0}, compute(p), "held while the peer's in-flight update may be in transit")
			}
			clk.SetTime(start.Add(tc.heldFor))
			assert.Equal(t, []float64{1.0, 0.9, 0.8, 0.7}, compute(p))
		})
	}
}

func TestComputeLimitHoldTracksTheHighestPeerBand(t *testing.T) {
	t.Parallel()
	start := time.Unix(1000, 0)
	clk := testclock.NewFakePassiveClock(start)
	peers := &fakePeers{queued: map[int]uint64{40: 2}, interval: 20 * time.Millisecond}
	p := New("", &ladder{})
	p.clock = clk
	p.SetPeerDemand(peers)
	require.Equal(t, []float64{1.0, 0.9, 0.8, 0}, compute(p))

	peers.queued = map[int]uint64{80: 1}
	require.Equal(t, []float64{1.0, 0, 0, 0}, compute(p))

	peers.queued = map[int]uint64{}
	clk.SetTime(start.Add(30 * time.Millisecond))
	assert.Equal(t, []float64{1.0, 0, 0, 0}, compute(p), "the hold covers the bands below the last peer band seen")
}

func ptr[T any](v T) *T { return &v }

func newHandle() fwkplugin.Handle {
	return fwkplugin.NewEppHandle(context.Background(), nil, fwkplugin.WithMetricsRecorder(prometheus.NewRegistry()))
}

func TestPolicyFactory(t *testing.T) {
	t.Parallel()
	handle := newHandle()
	handle.AddPlugin("static", usagelimits.NewConstPolicy("static", 0.6))
	handle.AddPlugin("not-a-policy", &fakePlugin{})

	p, err := PolicyFactory("held", fwkplugin.StrictDecoder([]byte(`{"delegate":"static"}`)), handle)
	require.NoError(t, err)
	policy := p.(*Policy)
	assert.Equal(t, fwkplugin.TypedName{Type: PolicyType, Name: "held"}, policy.TypedName())
	policy.SetPeerDemand(&fakePeers{queued: map[int]uint64{60: 1}})
	assert.Equal(t, []float64{0.6, 0.6, 0, 0}, compute(policy))

	_, err = PolicyFactory("held", fwkplugin.StrictDecoder([]byte(`{"delegate":"missing"}`)), handle)
	assert.ErrorContains(t, err, "delegate plugin not found: missing")
	_, err = PolicyFactory("held", fwkplugin.StrictDecoder([]byte(`{"delegate":"not-a-policy"}`)), handle)
	assert.ErrorContains(t, err, "does not implement flowcontrol.UsageLimitPolicy")
	p, err = PolicyFactory("delayed", fwkplugin.StrictDecoder([]byte(`{"delegate":"static","releaseDelay":"150ms"}`)), handle)
	require.NoError(t, err)
	assert.Equal(t, ptr(150*time.Millisecond), p.(*Policy).releaseDelay)
	_, err = PolicyFactory("held", fwkplugin.StrictDecoder([]byte(`{"releaseDelay":"-1s"}`)), handle)
	assert.ErrorContains(t, err, "releaseDelay must not be negative")
	_, err = PolicyFactory("held", fwkplugin.StrictDecoder([]byte(`{"delegat":"static"}`)), handle)
	assert.Error(t, err, "unknown fields are rejected")

	p, err = PolicyFactory("bare", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, []float64{1, 1, 1, 1}, compute(p.(*Policy)))
}

type fakePlugin struct{}

func (fakePlugin) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "fake", Name: "not-a-policy"}
}
