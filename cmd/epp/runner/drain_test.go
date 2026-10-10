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

package runner

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fcmocks "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts/mocks"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

// fakeServer stands in for a gRPC server runnable. Like GracefulStop, it returns
// only after its in-flight work finishes: stopped closes when its context ends,
// and the runnable returns once release is closed.
type fakeServer struct {
	stopped chan struct{}
	release chan struct{}
}

func newFakeServer() *fakeServer {
	return &fakeServer{stopped: make(chan struct{}), release: make(chan struct{})}
}

func (s *fakeServer) run(ctx context.Context) error {
	<-ctx.Done()
	close(s.stopped)
	<-s.release
	return nil
}

func closedWithin(ch <-chan struct{}, d time.Duration) bool {
	select {
	case <-ch:
		return true
	case <-time.After(d):
		return false
	}
}

func isClosed(ch <-chan struct{}) bool {
	select {
	case <-ch:
		return true
	default:
		return false
	}
}

func waitErr(t *testing.T, done <-chan error) error {
	t.Helper()
	select {
	case err := <-done:
		return err
	case <-time.After(5 * time.Second):
		t.Fatal("serveWithDrain did not return")
		return nil
	}
}

// A leader whose lease is lost keeps serving requests already routed to it, or
// arriving before the proxy sees the new leader, for the drain window. Its
// health server keeps answering liveness until ext_proc has finished its streams.
func TestServeWithDrainAfterLeaseLoss(t *testing.T) {
	const drain = 300 * time.Millisecond
	errLost := errors.New("leader election lost")
	lose := make(chan struct{})
	startManager := func(context.Context) error {
		<-lose
		return errLost
	}
	extProc, health := newFakeServer(), newFakeServer()
	close(health.release)
	draining, elected := &atomic.Bool{}, &atomic.Bool{}
	elected.Store(true)

	done := make(chan error, 1)
	go func() {
		done <- serveWithDrain(context.Background(), startManager, extProc.run, health.run, draining, elected, drain)
	}()
	close(lose)

	deadline := time.Now().Add(100 * time.Millisecond)
	for !draining.Load() && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if !draining.Load() {
		t.Fatal("readiness did not turn NotServing after the lease was lost")
	}
	if closedWithin(extProc.stopped, drain/2) {
		t.Fatal("ext_proc stopped before the drain window elapsed")
	}
	if !closedWithin(extProc.stopped, 2*drain) {
		t.Fatal("ext_proc was not stopped after the drain window")
	}
	if closedWithin(health.stopped, 100*time.Millisecond) {
		t.Fatal("health stopped while ext_proc was still finishing streams")
	}
	close(extProc.release)
	if !closedWithin(health.stopped, time.Second) {
		t.Fatal("health was not stopped after ext_proc returned")
	}
	if err := waitErr(t, done); !errors.Is(err, errLost) {
		t.Fatalf("serveWithDrain returned %v, want %v", err, errLost)
	}
}

// A manager that fails before this instance was ever elected has no traffic to
// drain: both servers stop at once and the error is returned.
func TestServeWithDrainManagerFailsBeforeElection(t *testing.T) {
	errStart := errors.New("cache sync failed")
	extProc, health := newFakeServer(), newFakeServer()
	close(extProc.release)
	close(health.release)
	draining, elected := &atomic.Bool{}, &atomic.Bool{}

	done := make(chan error, 1)
	go func() {
		done <- serveWithDrain(context.Background(), func(context.Context) error { return errStart },
			extProc.run, health.run, draining, elected, time.Minute)
	}()
	if err := waitErr(t, done); !errors.Is(err, errStart) {
		t.Fatalf("serveWithDrain returned %v, want %v", err, errStart)
	}
	if !isClosed(extProc.stopped) || !isClosed(health.stopped) {
		t.Fatal("servers still running after serveWithDrain returned")
	}
	if draining.Load() {
		t.Fatal("draining set by a manager failure before election")
	}
}

// With leader election disabled there is no lease to lose, so a manager failure
// stops both servers at once and the error is returned.
func TestServeWithDrainManagerFailsWithoutLeaderElection(t *testing.T) {
	errMgr := errors.New("controller failed")
	extProc, health := newFakeServer(), newFakeServer()
	close(extProc.release)
	close(health.release)
	draining := &atomic.Bool{}

	done := make(chan error, 1)
	go func() {
		done <- serveWithDrain(context.Background(), func(context.Context) error { return errMgr },
			extProc.run, health.run, draining, nil, time.Minute)
	}()
	if err := waitErr(t, done); !errors.Is(err, errMgr) {
		t.Fatalf("serveWithDrain returned %v, want %v", err, errMgr)
	}
	if !isClosed(extProc.stopped) || !isClosed(health.stopped) {
		t.Fatal("servers still running after serveWithDrain returned")
	}
	if draining.Load() {
		t.Fatal("draining set by a manager failure without leader election")
	}
}

// On SIGTERM the manager returns nil; the servers drain the same way.
func TestServeWithDrainOnSIGTERM(t *testing.T) {
	const drain = 300 * time.Millisecond
	ctx, sigterm := context.WithCancel(context.Background())
	startManager := func(c context.Context) error {
		<-c.Done()
		return nil
	}
	extProc, health := newFakeServer(), newFakeServer()
	close(health.release)
	draining, elected := &atomic.Bool{}, &atomic.Bool{}
	elected.Store(true)

	done := make(chan error, 1)
	go func() {
		done <- serveWithDrain(ctx, startManager, extProc.run, health.run, draining, elected, drain)
	}()
	sigterm()

	if closedWithin(extProc.stopped, drain/2) {
		t.Fatal("ext_proc stopped before the drain window elapsed")
	}
	if !draining.Load() {
		t.Fatal("readiness did not turn NotServing on SIGTERM")
	}
	if !closedWithin(extProc.stopped, 2*drain) {
		t.Fatal("ext_proc was not stopped after the drain window")
	}
	if closedWithin(health.stopped, 100*time.Millisecond) {
		t.Fatal("health stopped while ext_proc was still finishing streams")
	}
	close(extProc.release)
	if err := waitErr(t, done); err != nil {
		t.Fatalf("serveWithDrain returned %v, want nil", err)
	}
}

func TestFlowControlAdmissionDuringDrain(t *testing.T) {
	for _, tc := range []struct {
		name       string
		signal     bool
		managerErr error
	}{
		{name: "SIGTERM", signal: true},
		{name: "lease loss", managerErr: errors.New("leader election lost")},
	} {
		t.Run(tc.name, func(t *testing.T) {
			ctx, sigterm := context.WithCancel(context.Background())
			defer sigterm()
			r := NewRunner()
			admissionCtx, stopAdmission := context.WithCancel(context.WithoutCancel(ctx))
			defer stopAdmission()
			r.admissionCtx = admissionCtx
			opts := runserver.NewOptions()
			opts.PoolName = testPoolName
			opts.ConfigText = `apiVersion: llm-d.ai/v1
kind: EndpointPickerConfig
featureGates:
- flowControl
`
			rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
			require.NoError(t, err)
			ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
			eppConfig, err := r.parseConfigurationPhaseTwo(ctx, rawConfig, ds, opts.RefreshMetricsInterval)
			require.NoError(t, err)
			eppConfig.SaturationDetector = &fwkfcmocks.MockSaturationDetector{}
			candidates := &fcmocks.MockEndpointCandidates{Candidates: []fwkdl.Endpoint{fwkdl.NewEndpoint(nil, nil)}}
			_, admission, _, _ := r.initAdmissionControl(ctx, opts, eppConfig, candidates)
			admit := func(id string) error {
				requestCtx, cancel := context.WithTimeout(context.Background(), time.Second)
				defer cancel()
				return admission.Admit(requestCtx, &handlers.RequestContext{
					SchedulingRequest:        &fwksched.InferenceRequest{RequestID: id, FairnessID: "draining"},
					Request:                  &handlers.Request{Metadata: map[string]any{}},
					RequestSize:              100,
					RequestReceivedTimestamp: time.Now(),
					IncomingModelName:        "test-model",
				}, 0)
			}
			require.NoError(t, admit("before shutdown"))

			extProc, health := newFakeServer(), newFakeServer()
			close(health.release)
			t.Cleanup(func() {
				if !isClosed(extProc.release) {
					close(extProc.release)
				}
			})
			managerStopped := make(chan struct{})
			stopManager := make(chan struct{})
			startManager := func(context.Context) error {
				<-stopManager
				close(managerStopped)
				return tc.managerErr
			}
			elected := &atomic.Bool{}
			elected.Store(true)
			done := make(chan error, 1)
			go func() {
				err := serveWithDrain(ctx, startManager, extProc.run, health.run, &atomic.Bool{}, elected, 300*time.Millisecond)
				stopAdmission()
				done <- err
			}()
			if tc.signal {
				sigterm()
			}
			close(stopManager)
			require.True(t, closedWithin(managerStopped, time.Second))
			require.False(t, isClosed(extProc.stopped))
			require.NoError(t, admit("during drain"))
			require.True(t, closedWithin(extProc.stopped, time.Second))
			require.NoError(t, admit("while finishing streams"))
			close(extProc.release)
			require.ErrorIs(t, waitErr(t, done), tc.managerErr)
			require.ErrorContains(t, admit("after drain"), "flow controller is not running")
		})
	}
}
