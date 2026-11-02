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

package endpointhealth

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sourcemetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

var epoch = time.Date(2026, 10, 11, 18, 30, 44, 0, time.UTC)

// testClock is a manually advanced time source.
type testClock struct{ t time.Time }

func (c *testClock) now() time.Time          { return c.t }
func (c *testClock) advance(d time.Duration) { c.t = c.t.Add(d) }

func id(name string) fwkdl.ID {
	return k8stypes.NamespacedName{Name: name, Namespace: "default"}
}

// endpoint returns a candidate whose metrics-data-source failure run is status.
func endpoint(name string, status *fwkdl.PollStatus) fwksched.Endpoint {
	attrs := fwkdl.NewAttributes()
	if status != nil {
		attrs.Put(fwkdl.PollStatusDataKey(sourcemetrics.MetricsDataSourceType), status)
	}
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: id(name)}, fwkdl.NewMetrics(), attrs)
}

// endpointAt returns a candidate without a poll status, served at address.
func endpointAt(name, address string) fwksched.Endpoint {
	return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: id(name), Address: address, Port: "8000"},
		fwkdl.NewMetrics(), fwkdl.NewAttributes())
}

// failFrom delivers a 502 response to a request whose target endpoint is name
// at address. served is the endpoint the proxy reports it sent the request to,
// or empty when the proxy reports none.
func failFrom(f *EndpointHealthFilter, name, address, served string) {
	var reqMetadata map[string]any
	if served != "" {
		reqMetadata = map[string]any{metadata.DestinationEndpointNamespace: map[string]any{
			metadata.DestinationEndpointServedKey: served,
		}}
	}
	f.ResponseHeader(context.Background(), nil,
		&fwkrc.Response{Headers: map[string]string{reqcommon.PseudoHeaderStatus: "502"}, ReqMetadata: reqMetadata},
		&fwkdl.EndpointMetadata{ID: id(name), Address: address, Port: "8000"})
}

func failing(count int, since time.Time) *fwkdl.PollStatus {
	return &fwkdl.PollStatus{ConsecutiveFailures: count, FailingSince: since}
}

func names(endpoints []fwksched.Endpoint) []string {
	out := make([]string, 0, len(endpoints))
	for _, ep := range endpoints {
		out = append(out, ep.GetMetadata().ID.Name)
	}
	return out
}

func newTestFilter(t *testing.T, params parameters) (*EndpointHealthFilter, *testClock) {
	t.Helper()
	resetMetrics()
	t.Cleanup(resetMetrics)
	f, err := NewEndpointHealthFilter("health", params)
	require.NoError(t, err)
	clock := &testClock{t: epoch}
	f.now = clock.now
	return f, clock
}

func respond(f *EndpointHealthFilter, name string, status string) {
	f.ResponseHeader(context.Background(), nil,
		&fwkrc.Response{Headers: map[string]string{reqcommon.PseudoHeaderStatus: status}},
		&fwkdl.EndpointMetadata{ID: id(name)})
}

func intPtr(v int) *int { return &v }

func TestFactory(t *testing.T) {
	tests := []struct {
		name    string
		params  string
		wantErr string
		want    config
	}{
		{
			name:   "defaults",
			params: `{}`,
			want: config{
				pollSource:               sourcemetrics.MetricsDataSourceType,
				pollFailureThreshold:     3,
				pollFailureDuration:      10 * time.Second,
				responseFailureThreshold: 5,
				responseFailureCodes:     map[int]struct{}{502: {}, 503: {}},
				ejectionDuration:         30 * time.Second,
				maxEjectionDuration:      5 * time.Minute,
			},
		},
		{
			name: "all parameters",
			params: `{"pollSource": "vllm-metrics", "pollFailureThreshold": 5, "pollFailureDuration": "30s",
				"responseFailureThreshold": 2, "responseFailureCodes": [500, 502],
				"ejectionDuration": "1m", "maxEjectionDuration": "10m"}`,
			want: config{
				pollSource:               "vllm-metrics",
				pollFailureThreshold:     5,
				pollFailureDuration:      30 * time.Second,
				responseFailureThreshold: 2,
				responseFailureCodes:     map[int]struct{}{500: {}, 502: {}},
				ejectionDuration:         time.Minute,
				maxEjectionDuration:      10 * time.Minute,
			},
		},
		{
			name:   "response health disabled skips ejection validation",
			params: `{"responseFailureThreshold": 0, "ejectionDuration": "0s"}`,
			want: config{
				pollSource:           sourcemetrics.MetricsDataSourceType,
				pollFailureThreshold: 3,
				pollFailureDuration:  10 * time.Second,
				responseFailureCodes: map[int]struct{}{502: {}, 503: {}},
				maxEjectionDuration:  5 * time.Minute,
			},
		},
		{name: "malformed", params: `{"pollFailureThreshold": "three"}`, wantErr: "failed to parse"},
		{name: "malformed duration", params: `{"pollFailureDuration": "soon"}`, wantErr: "failed to parse"},
		{name: "negative poll threshold", params: `{"pollFailureThreshold": -1}`, wantErr: "pollFailureThreshold must be non-negative"},
		{name: "negative poll duration", params: `{"pollFailureDuration": "-1s"}`, wantErr: "pollFailureDuration must be non-negative"},
		{name: "negative response threshold", params: `{"responseFailureThreshold": -1}`, wantErr: "responseFailureThreshold must be non-negative"},
		{name: "both disabled", params: `{"pollFailureThreshold": 0, "responseFailureThreshold": 0}`, wantErr: "both zero"},
		{name: "non-5xx code", params: `{"responseFailureCodes": [429]}`, wantErr: "must be 5xx statuses, got 429"},
		{name: "empty codes", params: `{"responseFailureCodes": []}`, wantErr: "responseFailureCodes must not be empty"},
		{name: "zero ejection", params: `{"ejectionDuration": "0s"}`, wantErr: "ejectionDuration must be positive"},
		{name: "max below base", params: `{"ejectionDuration": "1m", "maxEjectionDuration": "30s"}`, wantErr: "must be at least ejectionDuration"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			p, err := Factory("health", json.NewDecoder(strings.NewReader(tt.params)), nil)
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			f := p.(*EndpointHealthFilter)
			assert.Equal(t, tt.want, f.tracker.cfg)
			assert.Equal(t, EndpointHealthFilterType, f.TypedName().Type)
			assert.Equal(t, "health", f.TypedName().Name)
			assert.Equal(t, fwkdl.PollStatusDataKey(tt.want.pollSource), f.pollStatusKey)
		})
	}
}

func TestFactoryNilParameters(t *testing.T) {
	p, err := Factory("", nil, nil)
	require.NoError(t, err)
	assert.Equal(t, EndpointHealthFilterType, p.TypedName().Name)
}

func TestConsumes(t *testing.T) {
	f, _ := newTestFilter(t, parameters{PollSource: "vllm-metrics"})
	assert.Equal(t, fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{fwkdl.PollStatusDataKey("vllm-metrics"): &fwkdl.PollStatus{}},
	}, f.Consumes())

	f, _ = newTestFilter(t, parameters{PollFailureThreshold: intPtr(0)})
	assert.Empty(t, f.Consumes().Required, "nothing is consumed with poll health disabled")
}

func TestPollHealth(t *testing.T) {
	tests := []struct {
		name   string
		status func(now time.Time) *fwkdl.PollStatus
		kept   bool
	}{
		{name: "never failed", status: func(time.Time) *fwkdl.PollStatus { return nil }, kept: true},
		{name: "recovered", status: func(time.Time) *fwkdl.PollStatus { return &fwkdl.PollStatus{} }, kept: true},
		{
			name:   "single failed poll",
			status: func(now time.Time) *fwkdl.PollStatus { return failing(1, now.Add(-time.Minute)) },
			kept:   true,
		},
		{
			name:   "too few failures over a long time",
			status: func(now time.Time) *fwkdl.PollStatus { return failing(2, now.Add(-time.Minute)) },
			kept:   true,
		},
		{
			name:   "burst of failures shorter than the duration",
			status: func(now time.Time) *fwkdl.PollStatus { return failing(100, now.Add(-9*time.Second)) },
			kept:   true,
		},
		{
			name:   "threshold failures over the duration",
			status: func(now time.Time) *fwkdl.PollStatus { return failing(3, now.Add(-10*time.Second)) },
			kept:   false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, clock := newTestFilter(t, parameters{})
			got := f.Filter(context.Background(), nil, []fwksched.Endpoint{
				endpoint("healthy", nil),
				endpoint("subject", tt.status(clock.now())),
			})
			if tt.kept {
				assert.Equal(t, []string{"healthy", "subject"}, names(got))
			} else {
				assert.Equal(t, []string{"healthy"}, names(got))
			}
		})
	}
}

func TestPollHealthReadsConfiguredSource(t *testing.T) {
	f, clock := newTestFilter(t, parameters{PollSource: "vllm-metrics"})
	attrs := fwkdl.NewAttributes()
	attrs.Put(fwkdl.PollStatusDataKey("vllm-metrics"), failing(3, clock.now().Add(-time.Minute)))
	custom := fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: id("custom")}, nil, attrs)
	otherSource := endpoint("other-source", failing(3, clock.now().Add(-time.Minute)))

	got := f.Filter(context.Background(), nil, []fwksched.Endpoint{endpoint("healthy", nil), custom, otherSource})
	assert.Equal(t, []string{"healthy", "other-source"}, names(got))
}

func TestPollHealthDisabled(t *testing.T) {
	f, clock := newTestFilter(t, parameters{PollFailureThreshold: intPtr(0)})
	got := f.Filter(context.Background(), nil, []fwksched.Endpoint{
		endpoint("healthy", nil),
		endpoint("failing", failing(1000, clock.now().Add(-time.Hour))),
	})
	assert.Equal(t, []string{"healthy", "failing"}, names(got))
}

func TestPollHealthReadmitsOnFirstSuccess(t *testing.T) {
	f, clock := newTestFilter(t, parameters{})
	since := clock.now()
	clock.advance(10 * time.Second)
	got := f.Filter(context.Background(), nil, []fwksched.Endpoint{endpoint("a", nil), endpoint("b", failing(200, since))})
	require.Equal(t, []string{"a"}, names(got))

	got = f.Filter(context.Background(), nil, []fwksched.Endpoint{endpoint("a", nil), endpoint("b", &fwkdl.PollStatus{})})
	assert.Equal(t, []string{"a", "b"}, names(got))
}

func TestFailOpen(t *testing.T) {
	f, clock := newTestFilter(t, parameters{})
	since := clock.now().Add(-time.Minute)
	candidates := []fwksched.Endpoint{endpoint("a", failing(5, since)), endpoint("b", failing(5, since))}

	got := f.Filter(context.Background(), nil, candidates)

	assert.Equal(t, []string{"a", "b"}, names(got))
	assert.InDelta(t, 1, counterValue(t, failOpens, "health"), 0)
	assert.InDelta(t, 2, gaugeValue(t, excludedEndpoints, "health"), 0, "fail-open does not readmit endpoints")
	assert.Empty(t, f.Filter(context.Background(), nil, nil))
}

func TestResponseEjection(t *testing.T) {
	f, clock := newTestFilter(t, parameters{})
	candidates := []fwksched.Endpoint{endpoint("a", nil), endpoint("b", nil)}
	filter := func() []string { return names(f.Filter(context.Background(), nil, candidates)) }

	for range 4 {
		respond(f, "b", "502")
	}
	require.Equal(t, []string{"a", "b"}, filter(), "four failures are below the threshold")

	respond(f, "b", "503")
	require.Equal(t, []string{"a"}, filter(), "the fifth consecutive failure ejects")

	clock.advance(30*time.Second - time.Millisecond)
	require.Equal(t, []string{"a"}, filter())
	clock.advance(time.Millisecond)
	require.Equal(t, []string{"a", "b"}, filter(), "the ejection ends after ejectionDuration")
}

func TestResponseEjectionCountsOnlyConsecutiveFailures(t *testing.T) {
	f, _ := newTestFilter(t, parameters{})
	candidates := []fwksched.Endpoint{endpoint("a", nil), endpoint("b", nil)}

	for range 4 {
		respond(f, "b", "502")
	}
	respond(f, "b", "200")
	for range 4 {
		respond(f, "b", "502")
	}
	for _, status := range []string{"500", "504", "429", "400"} {
		respond(f, "b", status)
	}
	assert.Equal(t, []string{"a", "b"}, names(f.Filter(context.Background(), nil, candidates)))
}

func TestResponseEjectionBackoff(t *testing.T) {
	f, clock := newTestFilter(t, parameters{
		EjectionDuration:    durationPtr(time.Minute),
		MaxEjectionDuration: durationPtr(150 * time.Second),
	})
	candidates := []fwksched.Endpoint{endpoint("a", nil), endpoint("b", nil)}
	excludedFor := func() time.Duration {
		t.Helper()
		eject := func() {
			for range 5 {
				respond(f, "b", "502")
			}
		}
		eject()
		start := clock.now()
		for len(f.Filter(context.Background(), nil, candidates)) < len(candidates) {
			clock.advance(time.Second)
		}
		return clock.now().Sub(start)
	}

	assert.Equal(t, time.Minute, excludedFor())
	assert.Equal(t, 2*time.Minute, excludedFor())
	assert.Equal(t, 150*time.Second, excludedFor(), "capped at maxEjectionDuration")

	respond(f, "b", "200")
	assert.Equal(t, time.Minute, excludedFor(), "a successful response resets the backoff")
}

func TestResponsesDuringEjectionAreIgnored(t *testing.T) {
	f, clock := newTestFilter(t, parameters{})
	candidates := []fwksched.Endpoint{endpoint("a", nil), endpoint("b", nil)}
	for range 5 {
		respond(f, "b", "502")
	}
	clock.advance(15 * time.Second)
	for range 20 {
		respond(f, "b", "502")
	}
	clock.advance(15 * time.Second)
	require.Equal(t, []string{"a", "b"}, names(f.Filter(context.Background(), nil, candidates)))

	for range 4 {
		respond(f, "b", "502")
	}
	assert.Equal(t, []string{"a", "b"}, names(f.Filter(context.Background(), nil, candidates)),
		"failures during the ejection do not count toward the next one")
}

func TestSuccessDuringEjectionKeepsBackoff(t *testing.T) {
	f, clock := newTestFilter(t, parameters{EjectionDuration: durationPtr(time.Minute)})
	candidates := []fwksched.Endpoint{endpoint("a", nil), endpoint("b", nil)}
	eject := func() {
		for range 5 {
			respond(f, "b", "502")
		}
	}
	eject()
	respond(f, "b", "200")
	clock.advance(time.Minute)
	require.Equal(t, []string{"a", "b"}, names(f.Filter(context.Background(), nil, candidates)))

	eject()
	clock.advance(time.Minute)
	assert.Equal(t, []string{"a"}, names(f.Filter(context.Background(), nil, candidates)),
		"a success during the ejection does not reset the backoff")
	clock.advance(time.Minute)
	assert.Equal(t, []string{"a", "b"}, names(f.Filter(context.Background(), nil, candidates)))
}

func TestResponsesServedByAnotherEndpointAreIgnored(t *testing.T) {
	f, _ := newTestFilter(t, parameters{})
	candidates := []fwksched.Endpoint{endpointAt("a", "10.0.0.1"), endpointAt("b", "10.0.0.2")}
	for range 5 {
		failFrom(f, "a", "10.0.0.1", "10.0.0.2:8000")
	}
	assert.Equal(t, []string{"a", "b"}, names(f.Filter(context.Background(), nil, candidates)),
		"failures served by another endpoint do not count toward the target")

	for range 5 {
		failFrom(f, "a", "10.0.0.1", "10.0.0.1:8000")
	}
	assert.Equal(t, []string{"b"}, names(f.Filter(context.Background(), nil, candidates)))
}

func TestRecreatedEndpointStartsHealthy(t *testing.T) {
	f, _ := newTestFilter(t, parameters{})
	for range 5 {
		failFrom(f, "b", "10.0.0.2", "")
	}
	require.Equal(t, []string{"a"}, names(f.Filter(context.Background(), nil,
		[]fwksched.Endpoint{endpointAt("a", "10.0.0.1"), endpointAt("b", "10.0.0.2")})))

	recreated := []fwksched.Endpoint{endpointAt("a", "10.0.0.1"), endpointAt("b", "10.0.0.3")}
	assert.Equal(t, []string{"a", "b"}, names(f.Filter(context.Background(), nil, recreated)),
		"an endpoint recreated at a new address does not inherit the ejection")
	assert.InDelta(t, 0, gaugeValue(t, excludedEndpoints, "health"), 0)

	for range 5 {
		failFrom(f, "b", "10.0.0.2", "")
	}
	assert.Equal(t, []string{"a", "b"}, names(f.Filter(context.Background(), nil, recreated)),
		"failed responses of the previous instance do not count toward the new one")
}

func TestResponseStatusHeader(t *testing.T) {
	tests := []struct {
		name    string
		headers map[string]string
		code    int
		ok      bool
	}{
		{name: "pseudo-header", headers: map[string]string{":status": "502"}, code: 502, ok: true},
		{name: "bare name", headers: map[string]string{"status": "503"}, code: 503, ok: true},
		{name: "pseudo-header wins", headers: map[string]string{":status": "200", "status": "502"}, code: 200, ok: true},
		{name: "missing", headers: map[string]string{"content-type": "application/json"}},
		{name: "not a number", headers: map[string]string{":status": "bad"}},
		{name: "nil", headers: nil},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, ok := responseStatus(tt.headers)
			assert.Equal(t, tt.ok, ok)
			assert.Equal(t, tt.code, code)
		})
	}
}

func TestResponseHealthDisabled(t *testing.T) {
	f, _ := newTestFilter(t, parameters{ResponseFailureThreshold: intPtr(0)})
	for range 100 {
		respond(f, "b", "502")
	}
	f.ResponseHeader(context.Background(), nil, nil, &fwkdl.EndpointMetadata{ID: id("b")})
	f.ResponseHeader(context.Background(), nil, &fwkrc.Response{}, nil)
	assert.Equal(t, []string{"a", "b"},
		names(f.Filter(context.Background(), nil, []fwksched.Endpoint{endpoint("a", nil), endpoint("b", nil)})))
}

func TestPollAndResponseExclusionsCombine(t *testing.T) {
	f, clock := newTestFilter(t, parameters{})
	since := clock.now()
	clock.advance(10 * time.Second)
	for range 5 {
		respond(f, "b", "502")
	}
	candidates := func(status *fwkdl.PollStatus) []fwksched.Endpoint {
		return []fwksched.Endpoint{endpoint("a", nil), endpoint("b", status)}
	}

	require.Equal(t, []string{"a"}, names(f.Filter(context.Background(), nil, candidates(failing(200, since)))))
	assert.InDelta(t, 1, gaugeValue(t, excludedEndpoints, "health"), 0)

	// The ejection expires while polls still fail: the endpoint stays out.
	clock.advance(30 * time.Second)
	require.Equal(t, []string{"a"}, names(f.Filter(context.Background(), nil, candidates(failing(800, since)))))

	// Polls recover: the endpoint is readmitted.
	require.Equal(t, []string{"a", "b"}, names(f.Filter(context.Background(), nil, candidates(&fwkdl.PollStatus{}))))
	assert.InDelta(t, 0, gaugeValue(t, excludedEndpoints, "health"), 0)
	assert.InDelta(t, 1, counterValue(t, ejections, "health", reasonPoll), 0)
	assert.InDelta(t, 1, counterValue(t, ejections, "health", reasonResponse), 0)
	assert.InDelta(t, 1, counterValue(t, readmissions, "health", reasonPoll), 0)
	assert.InDelta(t, 1, counterValue(t, readmissions, "health", reasonResponse), 0)
}

// TestTransitionsLogOnceAtInfo confirms each state transition logs one INFO
// line and that evaluations that change nothing log nothing at INFO.
func TestTransitionsLogOnceAtInfo(t *testing.T) {
	var lines []string
	logger := funcr.New(func(_, args string) { lines = append(lines, args) }, funcr.Options{})
	ctx := log.IntoContext(context.Background(), logger)
	f, clock := newTestFilter(t, parameters{})
	since := clock.now()
	clock.advance(10 * time.Second)
	filter := func(b *fwkdl.PollStatus) {
		for range 3 {
			f.Filter(ctx, nil, []fwksched.Endpoint{endpoint("a", nil), endpoint("b", b)})
		}
	}

	filter(failing(200, since))
	require.Len(t, lines, 1)
	assert.Contains(t, lines[0], "marking it unhealthy")

	filter(&fwkdl.PollStatus{})
	require.Len(t, lines, 2)
	assert.Contains(t, lines[1], "polls recovered")

	for range 8 {
		f.ResponseHeader(ctx, nil, &fwkrc.Response{Headers: map[string]string{reqcommon.PseudoHeaderStatus: "502"}},
			&fwkdl.EndpointMetadata{ID: id("b")})
	}
	require.Len(t, lines, 3)
	assert.Contains(t, lines[2], "ejecting it")

	clock.advance(time.Minute)
	filter(nil)
	require.Len(t, lines, 4)
	assert.Contains(t, lines[3], "ejection expired")

	// The only candidate turns unhealthy: its transition is logged, and the
	// fail-open of this and later calls is not logged at INFO.
	f.Filter(ctx, nil, []fwksched.Endpoint{endpoint("a", failing(200, since))})
	require.Len(t, lines, 5)
	f.Filter(ctx, nil, []fwksched.Endpoint{endpoint("a", failing(200, since))})
	assert.Len(t, lines, 5)
}

func TestPrune(t *testing.T) {
	f, clock := newTestFilter(t, parameters{})
	since := clock.now().Add(-time.Minute)
	f.Filter(context.Background(), nil, []fwksched.Endpoint{
		endpoint("a", nil), endpoint("b", failing(5, since)), endpoint("c", failing(5, since)),
	})
	require.InDelta(t, 2, gaugeValue(t, excludedEndpoints, "health"), 0)

	f.tracker.prune([]fwkdl.ID{id("a"), id("b")})

	assert.InDelta(t, 1, gaugeValue(t, excludedEndpoints, "health"), 0)
	assert.Len(t, f.tracker.endpoints, 2)
	assert.NotContains(t, f.tracker.endpoints, id("c"))
}

// TestIncidentReplay replays the failure pattern of an endpoint whose model
// server died behind a proxy: from the moment it died, every metrics poll and
// every request to it returned 502, while three other endpoints stayed healthy.
func TestIncidentReplay(t *testing.T) {
	const tick = 50 * time.Millisecond
	tests := []struct {
		name            string
		params          parameters
		requestsPerTick int
		wantEjectedBy   time.Duration
	}{
		{name: "response health ejects after five failed requests", params: parameters{}, requestsPerTick: 1, wantEjectedBy: 5 * tick},
		{name: "poll health ejects without traffic to the endpoint", params: parameters{ResponseFailureThreshold: intPtr(0)}, wantEjectedBy: 10 * time.Second},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			f, clock := newTestFilter(t, tt.params)
			died := clock.now()
			var status *fwkdl.PollStatus
			candidates := func() []fwksched.Endpoint {
				return []fwksched.Endpoint{endpoint("swan-a", nil), endpoint("swan-b", status), endpoint("swan-c", nil), endpoint("swan-d", nil)}
			}

			var ejectedAt time.Duration = -1
			for elapsed := time.Duration(0); elapsed <= 15*time.Second; elapsed += tick {
				clock.t = died.Add(elapsed)
				// The collector polls once per tick, and the poll fails.
				if status == nil {
					status = failing(0, clock.now())
				}
				status = failing(status.ConsecutiveFailures+1, status.FailingSince)

				kept := names(f.Filter(context.Background(), nil, candidates()))
				if !assert.Contains(t, []int{3, 4}, len(kept)) {
					return
				}
				if len(kept) == 3 {
					assert.NotContains(t, kept, "swan-b")
					if ejectedAt < 0 {
						ejectedAt = elapsed
					}
					continue
				}
				require.Negative(t, ejectedAt, "an ejected endpoint stays out while it keeps failing")
				for range tt.requestsPerTick {
					respond(f, "swan-b", "502")
				}
			}
			require.GreaterOrEqual(t, ejectedAt, time.Duration(0), "the endpoint was never ejected")
			assert.LessOrEqual(t, ejectedAt, tt.wantEjectedBy)

			// The model server comes back: the first successful poll readmits it.
			clock.advance(75 * time.Minute)
			status = &fwkdl.PollStatus{}
			assert.Len(t, f.Filter(context.Background(), nil, candidates()), 4)
		})
	}
}
