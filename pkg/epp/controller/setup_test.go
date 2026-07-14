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

package controller

import (
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/config"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	"github.com/llm-d/llm-d-router/pkg/common"
)

// capturingManager wraps a real manager but records the runnables registered via
// Add, so a test can inspect the controller a reconciler's SetupWithManager
// builds (in particular its leader-election requirement).
type capturingManager struct {
	manager.Manager
	added []manager.Runnable
}

func (m *capturingManager) Add(r manager.Runnable) error {
	m.added = append(m.added, r)
	return nil
}

func newCapturingManager(t *testing.T) *capturingManager {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		t.Fatalf("add clientgo scheme: %v", err)
	}
	if err := v1.Install(scheme); err != nil {
		t.Fatalf("install v1: %v", err)
	}
	if err := v1alpha2.Install(scheme); err != nil {
		t.Fatalf("install v1alpha2: %v", err)
	}
	// A syntactically valid rest.Config is enough: the manager builds its cache
	// and client lazily and we never Start it, so nothing connects.
	mgr, err := manager.New(&rest.Config{Host: "http://127.0.0.1:1"}, manager.Options{
		Scheme:  scheme,
		Metrics: metricsserver.Options{BindAddress: "0"},
		// Controller names are validated in a process-global registry; each case
		// registers the same name (e.g. "pod"), so skip the check here.
		Controller: config.Controller{SkipNameValidation: ptr.To(true)},
	})
	if err != nil {
		t.Fatalf("manager.New: %v", err)
	}
	return &capturingManager{Manager: mgr}
}

// TestReconcilerRunOnNonLeaders verifies that every reconciler that populates the
// datastore honors RunOnNonLeaders: the controller it registers must need leader
// election exactly when RunOnNonLeaders is false. This guards the (identical,
// copy-pasted) wiring across the four reconcilers against a missed or inverted
// case.
func TestReconcilerRunOnNonLeaders(t *testing.T) {
	setups := []struct {
		name  string
		setup func(mgr ctrl.Manager, runOnNonLeaders bool) error
	}{
		{
			name: "PodReconciler",
			setup: func(mgr ctrl.Manager, r bool) error {
				return (&PodReconciler{RunOnNonLeaders: r}).SetupWithManager(mgr)
			},
		},
		{
			name: "InferencePoolReconciler",
			setup: func(mgr ctrl.Manager, r bool) error {
				return (&InferencePoolReconciler{RunOnNonLeaders: r}).SetupWithManager(mgr)
			},
		},
		{
			name: "InferenceObjectiveReconciler",
			setup: func(mgr ctrl.Manager, r bool) error {
				return (&InferenceObjectiveReconciler{PoolGKNN: common.GKNN{}, RunOnNonLeaders: r}).SetupWithManager(mgr)
			},
		},
		{
			name: "InferenceModelRewriteReconciler",
			setup: func(mgr ctrl.Manager, r bool) error {
				return (&InferenceModelRewriteReconciler{PoolGKNN: common.GKNN{}, RunOnNonLeaders: r}).SetupWithManager(mgr)
			},
		},
	}

	for _, s := range setups {
		for _, runOnNonLeaders := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/runOnNonLeaders=%v", s.name, runOnNonLeaders), func(t *testing.T) {
				mgr := newCapturingManager(t)
				if err := s.setup(mgr, runOnNonLeaders); err != nil {
					t.Fatalf("SetupWithManager: %v", err)
				}
				if len(mgr.added) != 1 {
					t.Fatalf("expected exactly 1 runnable registered, got %d", len(mgr.added))
				}
				ler, ok := mgr.added[0].(manager.LeaderElectionRunnable)
				if !ok {
					t.Fatalf("registered runnable is not a LeaderElectionRunnable")
				}
				// runOnNonLeaders=true means "run everywhere" → must NOT need leader election.
				want := !runOnNonLeaders
				if got := ler.NeedLeaderElection(); got != want {
					t.Errorf("NeedLeaderElection() = %v, want %v", got, want)
				}
			})
		}
	}
}
