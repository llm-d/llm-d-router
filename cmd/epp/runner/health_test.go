/*
Copyright 2025 The Kubernetes Authors.
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

package runner

import (
	"context"
	"errors"
	"net"
	"strings"
	"sync/atomic"
	"testing"
	"testing/synctest"
	"time"

	extProcPb "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/go-logr/logr/funcr"
	"github.com/go-logr/logr/testr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	healthPb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/keepalive"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
	fwknet "github.com/llm-d/llm-d-router/test/framework/net"
)

// Mock Datastore
type mockDatastore struct {
	datastore.Datastore
	hasSynced bool
	pool      *datalayer.EndpointPool
	err       error
}

func (m *mockDatastore) PoolHasSynced() bool {
	return m.hasSynced
}

func (m *mockDatastore) PoolGet() (*datalayer.EndpointPool, error) {
	return m.pool, m.err
}

// Mock AppProtocolSupporter
type mockSupporter struct {
	protocols []v1.AppProtocol
}

func (m *mockSupporter) Claims() fwkrh.Claims {
	return fwkrh.Claims{
		Protocols: m.protocols,
	}
}

type mockReadinessChecker struct {
	name string
	err  error
}

func (m *mockReadinessChecker) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "mock", Name: m.name}
}

func (m *mockReadinessChecker) CheckReady() error { return m.err }

type mockPlugin struct{}

func (*mockPlugin) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: "mock", Name: "ordinary"}
}

func TestPluginReadinessCheckers(t *testing.T) {
	want := &mockReadinessChecker{name: "readiness"}
	got := pluginReadinessCheckers([]fwkplugin.Plugin{&mockPlugin{}, want})
	if len(got) != 1 || got[0] != want {
		t.Fatalf("pluginReadinessCheckers() = %v, want [%v]", got, want)
	}
}

func TestHealthServer_Check(t *testing.T) {
	tests := []struct {
		name                  string
		leaderElectionEnabled bool
		isLeader              bool
		draining              bool
		hasSynced             bool
		pool                  *datalayer.EndpointPool
		poolErr               error
		supporters            []appProtocolSupporter
		readinessCheckers     []fwkplugin.ReadinessChecker
		service               string
		wantStatus            healthPb.HealthCheckResponse_ServingStatus
	}{
		{
			name:                  "LeaderElectionDisabled_Live_ProtocolMatches",
			leaderElectionEnabled: false,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			wantStatus:            healthPb.HealthCheckResponse_SERVING,
		},
		{
			name:                  "LeaderElectionDisabled_NotLive",
			leaderElectionEnabled: false,
			hasSynced:             false,
			wantStatus:            healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			name:                  "LeaderElectionDisabled_ProtocolMismatch",
			leaderElectionEnabled: false,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolH2C},
			supporters:            []appProtocolSupporter{&mockSupporter{protocols: []v1.AppProtocol{v1.AppProtocolHTTP}}},
			wantStatus:            healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			// File-discovery pools and K8s pools without an explicit AppProtocol
			// must not be constrained to HTTP -- a non-HTTP parser would otherwise
			// lock the EPP out of SERVING permanently.
			name:                  "LeaderElectionDisabled_EmptyAppProtocol_NoConstraint",
			leaderElectionEnabled: false,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{},
			supporters:            []appProtocolSupporter{&mockSupporter{protocols: []v1.AppProtocol{v1.AppProtocolH2C}}},
			wantStatus:            healthPb.HealthCheckResponse_SERVING,
		},
		{
			name:                  "LeaderElectionDisabled_PluginNotReady",
			leaderElectionEnabled: false,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			readinessCheckers: []fwkplugin.ReadinessChecker{
				&mockReadinessChecker{name: "unready", err: errors.New("initial sync pending")},
			},
			wantStatus: healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			name:                  "LeaderElectionDisabled_Liveness_IgnoresPluginReadiness",
			leaderElectionEnabled: false,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			readinessCheckers: []fwkplugin.ReadinessChecker{
				&mockReadinessChecker{name: "unready", err: errors.New("initial sync pending")},
			},
			service:    LivenessCheckService,
			wantStatus: healthPb.HealthCheckResponse_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_Liveness_AlwaysServing",
			leaderElectionEnabled: true,
			service:               LivenessCheckService,
			wantStatus:            healthPb.HealthCheckResponse_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_Readiness_AllPluginsReady",
			leaderElectionEnabled: true,
			isLeader:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			readinessCheckers: []fwkplugin.ReadinessChecker{
				&mockReadinessChecker{name: "first"},
				&mockReadinessChecker{name: "second"},
			},
			service:    ReadinessCheckService,
			wantStatus: healthPb.HealthCheckResponse_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_Readiness_OnePluginNotReady",
			leaderElectionEnabled: true,
			isLeader:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			readinessCheckers: []fwkplugin.ReadinessChecker{
				&mockReadinessChecker{name: "ready"},
				&mockReadinessChecker{name: "unready", err: errors.New("backend unavailable")},
			},
			service:    ReadinessCheckService,
			wantStatus: healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_Liveness_IgnoresPluginReadiness",
			leaderElectionEnabled: true,
			readinessCheckers: []fwkplugin.ReadinessChecker{
				&mockReadinessChecker{name: "unready", err: errors.New("initial sync pending")},
			},
			service:    LivenessCheckService,
			wantStatus: healthPb.HealthCheckResponse_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_Readiness_Live_Leader_ProtocolMatches",
			leaderElectionEnabled: true,
			isLeader:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			service:               ReadinessCheckService,
			wantStatus:            healthPb.HealthCheckResponse_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_Readiness_NotLive",
			leaderElectionEnabled: true,
			isLeader:              true,
			hasSynced:             false,
			service:               ReadinessCheckService,
			wantStatus:            healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_Readiness_NotLeader",
			leaderElectionEnabled: true,
			isLeader:              false,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			service:               ReadinessCheckService,
			wantStatus:            healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_Readiness_ProtocolMismatch",
			leaderElectionEnabled: true,
			isLeader:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolH2C},
			supporters:            []appProtocolSupporter{&mockSupporter{protocols: []v1.AppProtocol{v1.AppProtocolHTTP}}},
			service:               ReadinessCheckService,
			wantStatus:            healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_EmptyService_ReflectsReadiness_Serving",
			leaderElectionEnabled: true,
			isLeader:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			service:               "",
			wantStatus:            healthPb.HealthCheckResponse_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_EmptyService_ReflectsPluginReadiness",
			leaderElectionEnabled: true,
			isLeader:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			readinessCheckers: []fwkplugin.ReadinessChecker{
				&mockReadinessChecker{name: "unready", err: errors.New("initial sync pending")},
			},
			service:    "",
			wantStatus: healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_ExtProc_ReflectsReadiness_Serving",
			leaderElectionEnabled: true,
			isLeader:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			service:               extProcPb.ExternalProcessor_ServiceDesc.ServiceName,
			wantStatus:            healthPb.HealthCheckResponse_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_ExtProc_ReflectsPluginReadiness",
			leaderElectionEnabled: true,
			isLeader:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			readinessCheckers: []fwkplugin.ReadinessChecker{
				&mockReadinessChecker{name: "unready", err: errors.New("initial sync pending")},
			},
			service:    extProcPb.ExternalProcessor_ServiceDesc.ServiceName,
			wantStatus: healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			name:                  "LeaderElectionEnabled_UnknownService",
			leaderElectionEnabled: true,
			service:               "unknown",
			wantStatus:            healthPb.HealthCheckResponse_SERVICE_UNKNOWN,
		},
		{
			// Graceful drain: an otherwise-Ready leader reports NotServing on
			// readiness so Kubernetes drains it from the Service endpoints.
			name:                  "LeaderElectionEnabled_Draining_Readiness_NotServing",
			leaderElectionEnabled: true,
			isLeader:              true,
			draining:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			service:               ReadinessCheckService,
			wantStatus:            healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			// Liveness must stay Serving while draining so we are not restarted
			// mid-drain.
			name:                  "LeaderElectionEnabled_Draining_Liveness_StillServing",
			leaderElectionEnabled: true,
			isLeader:              true,
			draining:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			service:               LivenessCheckService,
			wantStatus:            healthPb.HealthCheckResponse_SERVING,
		},
		{
			// In all-active mode (no leader election), liveness must stay Serving
			// during drain so kubelet does not restart the pod mid-drain.
			name:                  "LeaderElectionDisabled_Draining_Liveness_StillServing",
			leaderElectionEnabled: false,
			draining:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			service:               LivenessCheckService,
			wantStatus:            healthPb.HealthCheckResponse_SERVING,
		},
		{
			// In all-active mode, readiness should still fail while draining so
			// traffic is removed from the endpoint during graceful shutdown.
			name:                  "LeaderElectionDisabled_Draining_Readiness_NotServing",
			leaderElectionEnabled: false,
			draining:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			service:               ReadinessCheckService,
			wantStatus:            healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			// Empty service name is used by load balancers and should report
			// non-liveness drain state in all-active mode.
			name:                  "LeaderElectionDisabled_Draining_EmptyService_NotServing",
			leaderElectionEnabled: false,
			draining:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			service:               "",
			wantStatus:            healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			// The ext_proc service check is what Envoy's cluster health check hits.
			name:                  "LeaderElectionEnabled_Draining_ExtProc_NotServing",
			leaderElectionEnabled: true,
			isLeader:              true,
			draining:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			service:               extProcPb.ExternalProcessor_ServiceDesc.ServiceName,
			wantStatus:            healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			// The empty service name is used by load balancers for overall health.
			name:                  "LeaderElectionEnabled_Draining_EmptyService_NotServing",
			leaderElectionEnabled: true,
			isLeader:              true,
			draining:              true,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			service:               "",
			wantStatus:            healthPb.HealthCheckResponse_NOT_SERVING,
		},
		{
			name:                  "MultipleSupporters_AllMatch",
			leaderElectionEnabled: false,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			supporters: []appProtocolSupporter{
				&mockSupporter{protocols: []v1.AppProtocol{v1.AppProtocolHTTP}},
				&mockSupporter{protocols: []v1.AppProtocol{v1.AppProtocolHTTP}},
			},
			wantStatus: healthPb.HealthCheckResponse_SERVING,
		},
		{
			name:                  "MultipleSupporters_OneMismatch",
			leaderElectionEnabled: false,
			hasSynced:             true,
			pool:                  &datalayer.EndpointPool{AppProtocol: v1.AppProtocolHTTP},
			supporters: []appProtocolSupporter{
				&mockSupporter{protocols: []v1.AppProtocol{v1.AppProtocolHTTP}},
				&mockSupporter{protocols: []v1.AppProtocol{v1.AppProtocolH2C}},
			},
			wantStatus: healthPb.HealthCheckResponse_NOT_SERVING,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger := testr.New(t)
			ds := &mockDatastore{
				hasSynced: tt.hasSynced,
				pool:      tt.pool,
				err:       tt.poolErr,
			}
			var isLeader atomic.Bool
			isLeader.Store(tt.isLeader)
			var draining atomic.Bool
			draining.Store(tt.draining)

			s := &healthServer{
				logger:                logger,
				datastore:             ds,
				isLeader:              &isLeader,
				leaderElectionEnabled: tt.leaderElectionEnabled,
				supporters:            tt.supporters,
				readinessCheckers:     tt.readinessCheckers,
				draining:              &draining,
			}

			resp, err := s.Check(t.Context(), &healthPb.HealthCheckRequest{Service: tt.service})
			if err != nil {
				t.Fatalf("Check failed: %v", err)
			}
			if resp.Status != tt.wantStatus {
				t.Errorf("Check() status = %v, want %v", resp.Status, tt.wantStatus)
			}
		})
	}
}

func TestHealthServer_StandbyLogging(t *testing.T) {
	for _, service := range []string{ReadinessCheckService, "", extProcPb.ExternalProcessor_ServiceDesc.ServiceName} {
		t.Run(service, func(t *testing.T) {
			for _, tt := range []struct {
				name          string
				notSynced     bool
				protocol      v1.AppProtocol
				pluginErr     error
				draining      bool
				verbosity     int
				wantErrorLogs int
				wantInfoLogs  int
			}{
				{name: "healthy standby at trace", verbosity: logutil.TRACE, wantInfoLogs: 1},
				{name: "healthy standby below trace", verbosity: logutil.DEBUG},
				{name: "datastore not synced", notSynced: true, wantErrorLogs: 1},
				{name: "plugin not ready", pluginErr: errors.New("initial sync pending"), wantErrorLogs: 1},
				{name: "protocol mismatch", protocol: v1.AppProtocolH2C, wantErrorLogs: 2},
				{name: "draining", draining: true, wantErrorLogs: 1},
			} {
				t.Run(tt.name, func(t *testing.T) {
					var errorLines []string
					var infoLines []string
					capture := funcr.New(func(prefix, args string) {
						if strings.Contains(args, `"error"=`) {
							errorLines = append(errorLines, prefix+" "+args)
						} else {
							infoLines = append(infoLines, prefix+" "+args)
						}
					}, funcr.Options{Verbosity: tt.verbosity})
					var isLeader, draining atomic.Bool
					draining.Store(tt.draining)
					s := &healthServer{
						logger: capture,
						datastore: &mockDatastore{
							hasSynced: !tt.notSynced,
							pool:      &datalayer.EndpointPool{AppProtocol: tt.protocol},
						},
						isLeader:              &isLeader,
						leaderElectionEnabled: true,
						supporters:            []appProtocolSupporter{&mockSupporter{protocols: []v1.AppProtocol{v1.AppProtocolHTTP}}},
						readinessCheckers:     []fwkplugin.ReadinessChecker{&mockReadinessChecker{name: "test", err: tt.pluginErr}},
						draining:              &draining,
					}

					resp, err := s.Check(context.Background(), &healthPb.HealthCheckRequest{Service: service})
					if err != nil {
						t.Fatalf("Check failed: %v", err)
					}
					if resp.Status != healthPb.HealthCheckResponse_NOT_SERVING {
						t.Errorf("Check() status = %v, want NOT_SERVING", resp.Status)
					}
					if len(errorLines) != tt.wantErrorLogs {
						t.Errorf("got %d error logs, want %d: %v", len(errorLines), tt.wantErrorLogs, errorLines)
					}
					if len(infoLines) != tt.wantInfoLogs {
						t.Errorf("got %d info logs, want %d: %v", len(infoLines), tt.wantInfoLogs, infoLines)
					}
				})
			}
		})
	}
}

// Envoy health-checks on a reused connection and sends keepalive pings every 15s with no
// active stream, so the server must accept pings that frequent without sending GOAWAY.
func TestHealthServerKeepsIdleConnectionAtEnvoyKeepaliveInterval(t *testing.T) {
	t.Parallel()

	synctest.Test(t, func(t *testing.T) {
		// An in-memory listener keeps all I/O inside the bubble; goroutines blocked on a real
		// socket would stop the fake clock from advancing.
		lis := bufconn.Listen(1 << 16)
		var isLeader atomic.Bool
		isLeader.Store(true)
		srv := newHealthGRPCServer(testr.New(t), &mockDatastore{}, &isLeader, nil, false, nil, nil)
		serveDone := make(chan struct{})
		go func() {
			defer close(serveDone)
			_ = srv.Serve(lis)
		}()
		defer func() {
			srv.Stop()
			<-serveDone
		}()

		// Envoy's interval; a 10s client interval would sit on the server's 10s MinTime and could strike.
		conn, err := grpc.NewClient("passthrough:///pipe",
			grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
				return lis.DialContext(ctx)
			}),
			grpc.WithTransportCredentials(insecure.NewCredentials()),
			grpc.WithKeepaliveParams(keepalive.ClientParameters{Time: 15 * time.Second, Timeout: 3 * time.Second, PermitWithoutStream: true}))
		if err != nil {
			t.Fatalf("NewClient: %v", err)
		}
		defer func() { _ = conn.Close() }()

		checkCtx, checkCancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer checkCancel()
		if _, err := healthPb.NewHealthClient(conn).Check(checkCtx, &healthPb.HealthCheckRequest{}, grpc.WaitForReady(true)); err != nil {
			t.Fatalf("Check failed: %v (code %v)", err, status.Code(err))
		}
		// grpc-go updates the picker before the connectivity state manager, so a fast
		// loopback Check can return while GetState still reports CONNECTING.
		for state := conn.GetState(); state != connectivity.Ready; state = conn.GetState() {
			if !conn.WaitForStateChange(checkCtx, state) {
				t.Fatalf("connection state after Check = %v, want Ready", conn.GetState())
			}
		}

		// With the default enforcement policy the server sends GOAWAY after the third ping (about 45s).
		waitCtx, waitCancel := context.WithTimeout(context.Background(), 70*time.Second)
		defer waitCancel()
		start := time.Now()
		if conn.WaitForStateChange(waitCtx, connectivity.Ready) {
			t.Fatalf("idle connection left Ready after %v (state %v): server rejected 15s keepalive pings with GOAWAY too_many_pings",
				time.Since(start).Round(time.Second), conn.GetState())
		}
	})
}

// TestExtProcReadiness_FollowsServing checks that readiness is NOT_SERVING until
// the ext_proc server is serving, and drops again once it returns.
func TestExtProcReadiness_FollowsServing(t *testing.T) {
	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = lis.Close() })
	sr := runserver.NewDefaultExtProcServerRunner()
	sr.SecureServing = false
	sr.GrpcListener = lis
	sr.Datastore = datastore.NewDatastore(t.Context(), nil)

	var isLeader atomic.Bool
	isLeader.Store(true)
	s := &healthServer{
		logger:            testr.New(t),
		datastore:         &mockDatastore{hasSynced: true, pool: &datalayer.EndpointPool{}},
		isLeader:          &isLeader,
		readinessCheckers: []fwkplugin.ReadinessChecker{extProcReadiness{sr}},
	}
	check := func(service string) (healthPb.HealthCheckResponse_ServingStatus, error) {
		resp, err := s.Check(t.Context(), &healthPb.HealthCheckRequest{Service: service})
		return resp.GetStatus(), err
	}
	requireStatus := func(service string, want healthPb.HealthCheckResponse_ServingStatus, msg string) {
		got, err := check(service)
		require.NoError(t, err)
		require.Equal(t, want, got, msg)
	}

	requireStatus(ReadinessCheckService, healthPb.HealthCheckResponse_NOT_SERVING, "readiness before the server starts")
	requireStatus(LivenessCheckService, healthPb.HealthCheckResponse_SERVING, "liveness before the server starts")

	ctx, cancel := context.WithCancel(t.Context())
	t.Cleanup(cancel)
	done := make(chan error, 1)
	go func() { done <- sr.AsRunnable(logutil.NewTestLogger()).Start(ctx) }()
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		got, err := check(ReadinessCheckService)
		assert.NoError(c, err)
		assert.Equal(c, healthPb.HealthCheckResponse_SERVING, got)
	}, 10*time.Second, 10*time.Millisecond, "readiness never became SERVING while the server is serving")

	cancel()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("server did not return after cancel")
	}
	requireStatus(ReadinessCheckService, healthPb.HealthCheckResponse_NOT_SERVING, "readiness after the server returned")
}

// healthStatus asks the health server at addr for the status of service.
func healthStatus(ctx context.Context, addr, service string) (healthPb.HealthCheckResponse_ServingStatus, error) {
	cc, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return healthPb.HealthCheckResponse_UNKNOWN, err
	}
	defer cc.Close()
	ctx, cancel := context.WithTimeout(ctx, time.Second)
	defer cancel()
	resp, err := healthPb.NewHealthClient(cc).Check(ctx, &healthPb.HealthCheckRequest{Service: service})
	return resp.GetStatus(), err
}

// TestSetup_ReadinessWaitsForExtProc checks that the health server built by
// setup() reports NOT_SERVING until the ext_proc server is serving.
func TestSetup_ReadinessWaitsForExtProc(t *testing.T) {
	ctx := t.Context()
	modelServer, _ := newDrainTestBackend(t, &atomic.Bool{})
	opts := newDrainTestOptions(modelServer)
	r, _ := setupDrainTestRunner(ctx, t, opts)

	healthListener, err := fwknet.ReserveListener()
	require.NoError(t, err)
	go func() { _ = r.healthGRPCServer.Serve(healthListener) }()
	t.Cleanup(r.healthGRPCServer.Stop)
	healthAddr := healthListener.Addr().String()

	for _, service := range []string{ReadinessCheckService, "", extProcPb.ExternalProcessor_ServiceDesc.ServiceName} {
		got, err := healthStatus(ctx, healthAddr, service)
		require.NoError(t, err)
		require.Equal(t, healthPb.HealthCheckResponse_NOT_SERVING, got, "service %q before ext_proc starts", service)
	}

	grpcListener, err := fwknet.ReserveListener()
	require.NoError(t, err)
	r.serverRunner.GrpcListener = grpcListener
	serveCtx, cancel := context.WithCancel(ctx)
	done := make(chan error, 1)
	go func() { done <- r.serverRunner.AsRunnable(logutil.NewTestLogger()).Start(serveCtx) }()
	t.Cleanup(func() {
		cancel()
		<-done
	})

	require.EventuallyWithT(t, func(c *assert.CollectT) {
		got, err := healthStatus(ctx, healthAddr, ReadinessCheckService)
		assert.NoError(c, err)
		assert.Equal(c, healthPb.HealthCheckResponse_SERVING, got)
	}, 10*time.Second, 20*time.Millisecond, "readiness never became SERVING once ext_proc served")
}
