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

package controller

import (
	"context"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	testutil "github.com/llm-d/llm-d-router/pkg/epp/util/testing"
)

type recordingNotifier struct {
	upserts []fwkdl.PeerMetadata
	deletes []types.NamespacedName
}

func (n *recordingNotifier) Upsert(peer *fwkdl.PeerMetadata) {
	n.upserts = append(n.upserts, *peer)
}

func (n *recordingNotifier) Delete(id types.NamespacedName) {
	n.deletes = append(n.deletes, id)
}

func (n *recordingNotifier) reset() {
	n.upserts = nil
	n.deletes = nil
}

func readyPod(name, ns, ip string, lbls map[string]string) *corev1.Pod { //nolint:unparam
	return testutil.MakePod(name).
		Namespace(ns).
		ReadyCondition().
		IP(ip).
		Labels(lbls).
		Complete().
		ObjRef()
}

// TestEPPPeerReconciler covers:
//
//   - a ready pod matching the selector is upserted
//   - this replica's own pod is excluded by IP
//   - a no-change reconcile emits no events
//   - a deleted pod is removed
func TestEPPPeerReconciler(t *testing.T) {
	const (
		ns         = "test-ns"
		selfIP     = "10.0.0.1"
		peerIP     = "10.0.0.2"
		port       = "9010"
		labelKey   = "app"
		labelValue = "my-epp"
	)

	peerLabels := map[string]string{labelKey: labelValue}

	selfPod := readyPod("epp-0", ns, selfIP, peerLabels)
	peerPod := readyPod("epp-1", ns, peerIP, peerLabels)
	notReadyPod := testutil.MakePod("epp-notready").
		Namespace(ns).
		IP("10.0.0.3").
		Labels(peerLabels).
		Complete().
		ObjRef()
	wrongLabelPod := readyPod("epp-other", ns, "10.0.0.4", map[string]string{"app": "other"})

	scheme := runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		t.Fatalf("add scheme: %v", err)
	}

	fc := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(selfPod, peerPod, notReadyPod, wrongLabelPod).
		Build()

	notifier := &recordingNotifier{}
	firstReconcileCalled := false

	r := &EPPPeerReconciler{
		Reader:           fc,
		Notifier:         notifier,
		Selector:         labels.SelectorFromSet(peerLabels),
		Namespace:        ns,
		Port:             port,
		SelfAddress:      selfIP,
		OnFirstReconcile: func() { firstReconcileCalled = true },
	}

	ctx := context.Background()

	// --- First reconcile: should upsert epp-1 only ---
	if _, err := r.Reconcile(ctx, ctrl.Request{}); err != nil {
		t.Fatalf("first reconcile: %v", err)
	}
	if !firstReconcileCalled {
		t.Error("OnFirstReconcile was not called")
	}
	if len(notifier.upserts) != 1 {
		t.Fatalf("expected 1 upsert, got %d", len(notifier.upserts))
	}
	want := fwkdl.PeerMetadata{
		ID:      types.NamespacedName{Namespace: ns, Name: "epp-1"},
		Address: peerIP,
		Port:    port,
	}
	if notifier.upserts[0] != want {
		t.Errorf("upsert[0] = %+v, want %+v", notifier.upserts[0], want)
	}
	if len(notifier.deletes) != 0 {
		t.Errorf("expected 0 deletes, got %d", len(notifier.deletes))
	}

	// --- Second reconcile (no changes): no events ---
	notifier.reset()
	if _, err := r.Reconcile(ctx, ctrl.Request{}); err != nil {
		t.Fatalf("second reconcile: %v", err)
	}
	if len(notifier.upserts) != 0 {
		t.Errorf("expected 0 upserts on no-change reconcile, got %d", len(notifier.upserts))
	}
	if len(notifier.deletes) != 0 {
		t.Errorf("expected 0 deletes on no-change reconcile, got %d", len(notifier.deletes))
	}

	// --- Third reconcile after removing the peer pod: should delete epp-1 ---
	if err := fc.Delete(ctx, peerPod); err != nil {
		t.Fatalf("delete peer pod: %v", err)
	}
	notifier.reset()
	if _, err := r.Reconcile(ctx, ctrl.Request{}); err != nil {
		t.Fatalf("third reconcile: %v", err)
	}
	if len(notifier.upserts) != 0 {
		t.Errorf("expected 0 upserts after delete, got %d", len(notifier.upserts))
	}
	if len(notifier.deletes) != 1 {
		t.Fatalf("expected 1 delete, got %d", len(notifier.deletes))
	}
	wantDelete := types.NamespacedName{Namespace: ns, Name: "epp-1"}
	if notifier.deletes[0] != wantDelete {
		t.Errorf("delete[0] = %v, want %v", notifier.deletes[0], wantDelete)
	}
}
