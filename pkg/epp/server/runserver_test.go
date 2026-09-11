/*
Copyright 2025 The Kubernetes Authors.

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

package server_test

import (
	"crypto/tls"
	"reflect"
	"testing"
	"time"

	"github.com/spf13/pflag"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/manager"

	"github.com/llm-d/llm-d-router/pkg/common"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
	"github.com/llm-d/llm-d-router/pkg/epp/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/server"
)

func TestRunnable(t *testing.T) {
	// Make sure AsRunnable() does not use leader election.
	runner := server.NewDefaultExtProcServerRunner().AsRunnable(logutil.NewTestLogger())
	r, ok := runner.(manager.LeaderElectionRunnable)
	if !ok {
		t.Fatal("runner is not LeaderElectionRunnable")
	}
	if r.NeedLeaderElection() {
		t.Error("runner returned NeedLeaderElection = true, expected false")
	}
}

// stubDatastore and stubBands are non-nil placeholders: the factory only stores these
// dependencies, so an embedded nil interface is enough to make the field observably set.
type stubDatastore struct{ datastore.Datastore }
type stubBands struct {
	contracts.PriorityBandControlPlane
}

// TestNewExtProcServerRunnerPopulatesEveryField fails the day a field is added to
// ExtProcServerRunner without being wired through the constructor -- the drift that let
// a new Options flag reach only one of the two hand-written composite literals.
func TestNewExtProcServerRunnerPopulatesEveryField(t *testing.T) {
	// Fields the caller sets after construction, not the factory.
	callerSet := map[string]bool{"GrpcListener": true, "EvictChannelLookup": true}

	opts := server.NewOptions()
	// AddFlags registers the logging flag set that Complete() dereferences.
	opts.AddFlags(pflag.NewFlagSet("t", pflag.ContinueOnError))
	opts.GRPCPort, opts.CertPath = 19002, "/tmp/certs"
	opts.SecureServing, opts.HealthChecking, opts.EnableCertReload = true, true, true
	opts.EnableGRPCStreamMetrics, opts.EmitEndpointScores = true, true
	opts.TLSMinVersion, opts.TLSCipherSuites = "VersionTLS13", []string{"TLS_AES_128_GCM_SHA256"}
	opts.RefreshPrometheusMetricsInterval, opts.MetricsStalenessThreshold = 7*time.Second, 11*time.Second
	require.NoError(t, opts.Complete()) // parses the TLS strings into the unexported fields
	// Set after Complete(): the *Str flags are empty, so Complete() leaves these alone.
	opts.GRPCMaxRecvMsgSize, opts.GRPCMaxSendMsgSize = 4<<20, 5<<20

	ds, sd := &stubDatastore{}, &fwkfcmocks.MockSaturationDetector{}
	director, registry, bands := &requestcontrol.Director{}, &handlers.ParserRegistry{}, &stubBands{}
	r := server.NewExtProcServerRunner(opts,
		common.GKNN{NamespacedName: types.NamespacedName{Name: "pool", Namespace: "ns"}},
		server.NewControllerConfig(true), ds, director, registry, sd, bands)

	v := reflect.ValueOf(*r)
	fields := reflect.VisibleFields(reflect.TypeOf(*r))
	require.NotEmpty(t, fields) // never vacuous
	for _, f := range fields {
		// FieldByIndex, not Field(i): VisibleFields flattens promoted fields.
		if !callerSet[f.Name] && v.FieldByIndex(f.Index).IsZero() {
			t.Errorf("%s left zero by NewExtProcServerRunner", f.Name)
		}
	}

	// IsZero cannot see two fields wired to each other's source, so pin every field that
	// shares its type with another one: the three ints, the two durations, and each
	// dependency. The bools cannot be told apart by value and are covered by
	// TestNewExtProcServerRunnerMapsEachBoolOption below.
	require.Equal(t, 4<<20, r.GRPCMaxRecvMsgSize)
	require.Equal(t, 5<<20, r.GRPCMaxSendMsgSize)
	require.Equal(t, 19002, r.GrpcPort)
	require.Equal(t, uint16(tls.VersionTLS13), r.TLSMinVersion)
	require.Equal(t, 7*time.Second, r.RefreshPrometheusMetricsInterval)
	require.Equal(t, 11*time.Second, r.MetricsStalenessThreshold)
	require.Same(t, ds, r.Datastore)
	require.Same(t, director, r.Director)
	require.Same(t, registry, r.ParserRegistry)
	require.Same(t, sd, r.SaturationDetector)
	require.Same(t, bands, r.PriorityBandControlPlane)
}

// boolOptions pairs each boolean Options field the factory reads with the runner field it
// must land in.
var boolOptions = []struct {
	name string
	set  func(*server.Options, bool)
	get  func(*server.ExtProcServerRunner) bool
}{
	{"SecureServing",
		func(o *server.Options, v bool) { o.SecureServing = v },
		func(r *server.ExtProcServerRunner) bool { return r.SecureServing }},
	{"HealthChecking",
		func(o *server.Options, v bool) { o.HealthChecking = v },
		func(r *server.ExtProcServerRunner) bool { return r.HealthChecking }},
	{"EnableCertReload",
		func(o *server.Options, v bool) { o.EnableCertReload = v },
		func(r *server.ExtProcServerRunner) bool { return r.EnableCertReload }},
	{"EnableGRPCStreamMetrics",
		func(o *server.Options, v bool) { o.EnableGRPCStreamMetrics = v },
		func(r *server.ExtProcServerRunner) bool { return r.EnableGRPCStreamMetrics }},
	{"EmitEndpointScores",
		func(o *server.Options, v bool) { o.EmitEndpointScores = v },
		func(r *server.ExtProcServerRunner) bool { return r.EmitEndpointScores }},
}

// TestNewExtProcServerRunnerMapsEachBoolOption sets one boolean option at a time and
// asserts exactly one runner field follows. Every bool holding the same value cannot
// distinguish a swapped pair, which is the wiring mistake the all-true sweep above misses.
func TestNewExtProcServerRunnerMapsEachBoolOption(t *testing.T) {
	for _, tc := range boolOptions {
		t.Run(tc.name, func(t *testing.T) {
			opts := server.NewOptions()
			for _, o := range boolOptions {
				o.set(opts, false) // NewOptions defaults SecureServing to true
			}
			tc.set(opts, true)

			r := server.NewExtProcServerRunner(opts, common.GKNN{},
				server.NewControllerConfig(true), nil, nil, nil, nil, nil)

			for _, o := range boolOptions {
				require.Equal(t, o.name == tc.name, o.get(r), "runner field %s", o.name)
			}
		})
	}
}
