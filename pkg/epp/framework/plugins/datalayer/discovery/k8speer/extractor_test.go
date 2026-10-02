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

package k8speer

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/statesync"
)

type testRegistrar struct {
	registration fwkdl.PendingRegistration
}

func (r *testRegistrar) Register(reg fwkdl.PendingRegistration) error {
	r.registration = reg
	return nil
}

func TestFactoryUsesHostnameWhenPodIPUnset(t *testing.T) {
	t.Setenv("NAMESPACE", "ns")
	t.Setenv("POD_IP", "")

	plugin, err := Factory("peer-disc", fwkplugin.StrictDecoder(
		json.RawMessage(`{"selector":"app=epp","port":"9002","namespace":"ns"}`)), nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	peerPlugin := plugin.(*Plugin)
	if peerPlugin.selfName == "" {
		t.Fatal("selfName is empty")
	}
	if peerPlugin.acceptsPod(makePod(peerPlugin.selfName, "ns", "10.0.0.1", map[string]string{"app": "epp"}, true)) {
		t.Fatal("plugin accepts self Pod when POD_IP is unset")
	}
}

func TestFactoryRejectsInvalidPort(t *testing.T) {
	t.Setenv("NAMESPACE", "ns")
	t.Setenv("POD_IP", "10.0.0.1")

	for _, port := range []string{"0", "65536", "invalid"} {
		t.Run(port, func(t *testing.T) {
			_, err := Factory("peer-disc", fwkplugin.StrictDecoder(json.RawMessage(
				`{"selector":"app=epp","port":"`+port+`","namespace":"ns"}`)), nil)
			if err == nil {
				t.Fatal("Factory error = nil, want invalid port error")
			}
		})
	}
}

func TestFactoryRejectsNamespaceMismatch(t *testing.T) {
	t.Setenv("NAMESPACE", "actual-ns")
	t.Setenv("POD_IP", "10.0.0.1")

	_, err := Factory("peer-disc", fwkplugin.StrictDecoder(
		json.RawMessage(`{"selector":"app=epp","port":"9002","namespace":"configured-ns"}`)), nil)
	if err == nil {
		t.Fatal("Factory error = nil, want namespace mismatch error")
	}
}

func TestPodExtractorFiltersAndTracksPeers(t *testing.T) {
	t.Setenv("NAMESPACE", "ns")
	t.Setenv("POD_IP", "10.0.0.1")
	plugin, err := Factory("peer-disc", fwkplugin.StrictDecoder(
		json.RawMessage(`{"selector":"app=epp","port":"9002","namespace":"ns"}`)), nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	peerPlugin := plugin.(*Plugin)

	var registrar testRegistrar
	if err := peerPlugin.RegisterDependencies(&registrar); err != nil {
		t.Fatalf("RegisterDependencies: %v", err)
	}
	extractor := registrar.registration.Extractor.(fwkdl.NotificationExtractor)

	ctx := t.Context()

	// Extract blocks on the unbuffered events channel until Start receives,
	// so run it in a goroutine and read the event it emits.
	extract := func(event fwkdl.NotificationEvent) peerEvent {
		errCh := make(chan error, 1)
		go func() { errCh <- extractor.Extract(ctx, event) }()
		ev := <-peerPlugin.events
		if err := <-errCh; err != nil {
			t.Fatalf("Extract: %v", err)
		}
		return ev
	}

	peer := makePod("peer", "ns", "10.0.0.2", map[string]string{"app": "epp"}, true)
	ev := extract(fwkdl.NotificationEvent{Type: fwkdl.EventAddOrUpdate, Object: toUnstructured(t, peer)})
	if ev.kind != fwkdl.EventAddOrUpdate || ev.peer == nil || ev.peer.Address != "10.0.0.2" {
		t.Fatalf("event = %+v, want upsert of 10.0.0.2", ev)
	}

	notReady := makePod("not-ready", "ns", "10.0.0.3", map[string]string{"app": "epp"}, false)
	ev = extract(fwkdl.NotificationEvent{Type: fwkdl.EventAddOrUpdate, Object: toUnstructured(t, notReady)})
	if ev.kind != fwkdl.EventDelete || ev.id.Name != "not-ready" {
		t.Fatalf("event = %+v, want delete of not-ready", ev)
	}

	ev = extract(fwkdl.NotificationEvent{Type: fwkdl.EventDelete, Object: toUnstructured(t, peer)})
	if ev.kind != fwkdl.EventDelete || ev.id.Name != "peer" {
		t.Fatalf("event = %+v, want delete of peer", ev)
	}

	unknown := fwkdl.NotificationEvent{Type: fwkdl.EventType(99), Object: toUnstructured(t, peer)}
	if err := extractor.Extract(ctx, unknown); err == nil {
		t.Fatal("Extract unknown event type: want error")
	}

	// Pods in other namespaces return without a handoff. Nothing reads the
	// events channel here, so a handoff would block until the deadline.
	shortCtx, cancel := context.WithTimeout(ctx, 100*time.Millisecond)
	defer cancel()
	other := makePod("other", "other-ns", "10.0.0.4", map[string]string{"app": "epp"}, true)
	for _, eventType := range []fwkdl.EventType{fwkdl.EventAddOrUpdate, fwkdl.EventDelete} {
		event := fwkdl.NotificationEvent{Type: eventType, Object: toUnstructured(t, other)}
		if err := extractor.Extract(shortCtx, event); err != nil {
			t.Fatalf("Extract pod in other namespace (type %v): %v", eventType, err)
		}
	}
}

func TestApplyPeerEvent(t *testing.T) {
	store := statesync.NewMemoryPeerStore()
	notifier := fwkdl.NewPeerNotifier(store)
	id := types.NamespacedName{Name: "epp-1", Namespace: "ns"}

	upsert := peerEvent{kind: fwkdl.EventAddOrUpdate, peer: &fwkdl.PeerMetadata{ID: id, Address: "10.0.0.2"}}
	if err := apply(notifier, upsert); err != nil {
		t.Fatalf("apply upsert: %v", err)
	}
	if got := store.Peers(); len(got) != 1 || got[0].Address != "10.0.0.2" {
		t.Fatalf("peers = %+v, want peer 10.0.0.2", got)
	}

	if err := apply(notifier, peerEvent{kind: fwkdl.EventDelete, id: id}); err != nil {
		t.Fatalf("apply delete: %v", err)
	}
	if got := store.Peers(); len(got) != 0 {
		t.Fatalf("peers after delete = %+v, want none", got)
	}

	if err := apply(notifier, peerEvent{kind: fwkdl.EventType(99)}); err == nil {
		t.Fatal("apply unknown kind: want error")
	}
}

func TestStartSignalsReadyWithoutPeers(t *testing.T) {
	t.Setenv("NAMESPACE", "ns")
	t.Setenv("POD_IP", "10.0.0.1")
	plugin, err := Factory("peer-disc", fwkplugin.StrictDecoder(
		json.RawMessage(`{"selector":"app=epp","port":"9002","namespace":"ns"}`)), nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	peerPlugin := plugin.(*Plugin)

	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() { errCh <- peerPlugin.Start(ctx, nil) }()

	select {
	case <-peerPlugin.Ready():
	case <-ctx.Done():
		t.Fatal("plugin did not become ready")
	}
	cancel()
	if err := <-errCh; err != nil {
		t.Fatalf("Start: %v", err)
	}
}

func makePod(name, namespace, ip string, labels map[string]string, ready bool) *corev1.Pod {
	status := corev1.ConditionFalse
	if ready {
		status = corev1.ConditionTrue
	}
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace, Labels: labels},
		Status: corev1.PodStatus{
			PodIP:      ip,
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: status}},
		},
	}
}

func toUnstructured(t *testing.T, pod *corev1.Pod) *unstructured.Unstructured {
	t.Helper()
	obj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(pod)
	if err != nil {
		t.Fatalf("convert pod: %v", err)
	}
	return &unstructured.Unstructured{Object: obj}
}
