/*
Copyright 2026 Google LLC.

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

package endpointslice

import (
	"context"
	"sync"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	discoveryv1 "k8s.io/api/discovery/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/utils/ptr"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// recorder is a DiscoveryNotifier that records calls and current state.
type recorder struct {
	mu      sync.Mutex
	state   map[types.NamespacedName]*fwkdl.EndpointMetadata
	upserts int
	deletes int
}

func newRecorder() *recorder {
	return &recorder{state: map[types.NamespacedName]*fwkdl.EndpointMetadata{}}
}

func (r *recorder) Upsert(m *fwkdl.EndpointMetadata) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.upserts++
	r.state[m.ID] = m
}

func (r *recorder) Delete(id types.NamespacedName) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.deletes++
	delete(r.state, id)
}

func (r *recorder) snapshot() map[types.NamespacedName]*fwkdl.EndpointMetadata {
	r.mu.Lock()
	defer r.mu.Unlock()
	out := make(map[types.NamespacedName]*fwkdl.EndpointMetadata, len(r.state))
	for k, v := range r.state {
		out[k] = v
	}
	return out
}

// ecpSlice builds an EndpointSlice exactly the way ECP's
// applyEndpointSliceInNamespace does for a Borneo runner (server_main.go):
// IPv6, one "http" port, hostname=runner, ready/serving hard-coded true, no
// targetRef/nodeName, and the exos/* labels and annotations.
func ecpSlice(runner, ipv6 string, ready bool, extraLabels map[string]string) *discoveryv1.EndpointSlice {
	l := map[string]string{
		discoveryv1.LabelServiceName: "glm52-server",
		"exos/runner-id":             runner,
		"exos/mode":                  "borneo",
	}
	for k, v := range extraLabels {
		l[k] = v
	}
	return &discoveryv1.EndpointSlice{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "glm52-server-" + runner + "-borneo",
			Namespace: "default",
			Labels:    l,
			Annotations: map[string]string{
				"exos/k8s-fqdn": "0-" + runner + ".uniserve-dev-jobs-pnatlix.borg.svc.cluster.local",
			},
		},
		AddressType: discoveryv1.AddressTypeIPv6,
		Ports:       []discoveryv1.EndpointPort{{Name: ptr.To("http"), Port: ptr.To[int32](9080), Protocol: ptr.To(corev1.ProtocolTCP)}},
		Endpoints: []discoveryv1.Endpoint{{
			Addresses:  []string{ipv6},
			Hostname:   ptr.To(runner),
			Conditions: discoveryv1.EndpointConditions{Ready: ptr.To(ready), Serving: ptr.To(ready)},
		}},
	}
}

func startPlugin(t *testing.T, cs *fake.Clientset, p Params) (*Discovery, *recorder, context.CancelFunc) {
	t.Helper()
	d, err := New("test", p, cs)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	rec := newRecorder()
	ctx, cancel := context.WithCancel(context.Background())
	errCh := make(chan error, 1)
	go func() { errCh <- d.Start(ctx, rec) }()
	select {
	case <-d.Ready():
	case err := <-errCh:
		t.Fatalf("Start returned before Ready: %v", err)
	case <-time.After(10 * time.Second):
		t.Fatal("plugin never became ready")
	}
	return d, rec, cancel
}

func waitFor(t *testing.T, what string, cond func() bool) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if cond() {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %s", what)
}

var exosParams = Params{
	Namespace:            "default",
	LabelSelector:        discoveryv1.LabelServiceName + "=glm52-server",
	PortName:             "http",
	LabelMappings:        map[string]string{"exos/role": "llm-d.ai/role"},
	StaticLabels:         map[string]string{"llm-d.ai/engine-type": "sglang"},
	AttributeAnnotations: []string{"exos/k8s-fqdn"},
}

func TestInitialSyncDiscoversECPSlices(t *testing.T) {
	cs := fake.NewClientset(
		ecpSlice("borneo-p0", "2606:ec80:10::10", true, map[string]string{"exos/role": "prefill"}),
		ecpSlice("borneo-d0", "2606:ec80:10::20", true, map[string]string{"exos/role": "decode"}),
		// A slice of another Service must be ignored by the selector.
		func() *discoveryv1.EndpointSlice {
			s := ecpSlice("other", "2606:ec80:10::99", true, nil)
			s.Labels[discoveryv1.LabelServiceName] = "something-else"
			s.Name = "something-else-abc"
			return s
		}(),
	)
	_, rec, cancel := startPlugin(t, cs, exosParams)
	defer cancel()

	got := rec.snapshot()
	if len(got) != 2 {
		t.Fatalf("got %d endpoints, want 2: %v", len(got), got)
	}
	d0 := got[types.NamespacedName{Namespace: "default", Name: "glm52-server.borneo-d0"}]
	if d0 == nil {
		t.Fatalf("decode endpoint missing: %v", got)
	}
	if d0.Address != "2606:ec80:10::20" || d0.Port != "9080" || d0.MetricsHost != "[2606:ec80:10::20]:9080" {
		t.Errorf("decode endpoint = %+v", d0)
	}
	if d0.Labels["llm-d.ai/role"] != "decode" || d0.Labels["llm-d.ai/engine-type"] != "sglang" ||
		d0.Labels["exos/runner-id"] != "borneo-d0" {
		t.Errorf("decode labels = %v", d0.Labels)
	}
	if d0.Attributes["exos/k8s-fqdn"] == "" {
		t.Errorf("fqdn attribute missing: %v", d0.Attributes)
	}
}

func TestReadinessAndDeletion(t *testing.T) {
	cs := fake.NewClientset(ecpSlice("borneo-d0", "2606:ec80:10::20", true, nil))
	_, rec, cancel := startPlugin(t, cs, exosParams)
	defer cancel()
	ctx := context.Background()
	id := types.NamespacedName{Namespace: "default", Name: "glm52-server.borneo-d0"}

	// ready=false (what ECP should write once model readiness is wired) removes it.
	s := ecpSlice("borneo-d0", "2606:ec80:10::20", false, nil)
	if _, err := cs.DiscoveryV1().EndpointSlices("default").Update(ctx, s, metav1.UpdateOptions{}); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "not-ready endpoint removed", func() bool { _, ok := rec.snapshot()[id]; return !ok })

	// Back to ready, with a new address (runner rescheduled): re-added with the new IP.
	s = ecpSlice("borneo-d0", "2606:ec80:10::21", true, nil)
	if _, err := cs.DiscoveryV1().EndpointSlices("default").Update(ctx, s, metav1.UpdateOptions{}); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "endpoint re-added with new IP", func() bool {
		m := rec.snapshot()[id]
		return m != nil && m.Address == "2606:ec80:10::21"
	})

	// Runner unlinked: ECP deletes the slice, the endpoint goes away.
	if err := cs.DiscoveryV1().EndpointSlices("default").Delete(ctx, s.Name, metav1.DeleteOptions{}); err != nil {
		t.Fatal(err)
	}
	waitFor(t, "endpoint deleted with slice", func() bool { return len(rec.snapshot()) == 0 })
}

func TestDuplicateAcrossSlicesReportedOnce(t *testing.T) {
	a := ecpSlice("borneo-d0", "2606:ec80:10::20", true, nil)
	b := ecpSlice("borneo-d0", "2606:ec80:10::20", true, nil)
	b.Name = "glm52-server-borneo-d0-replacement"
	cs := fake.NewClientset(a, b)
	_, rec, cancel := startPlugin(t, cs, exosParams)
	defer cancel()
	if n := len(rec.snapshot()); n != 1 {
		t.Fatalf("got %d endpoints, want 1", n)
	}
}

func TestUnchangedSliceDoesNotReUpsert(t *testing.T) {
	cs := fake.NewClientset(ecpSlice("borneo-d0", "2606:ec80:10::20", true, nil))
	_, rec, cancel := startPlugin(t, cs, exosParams)
	defer cancel()
	// ECP re-applies the identical slice periodically; that must not churn the datastore.
	s := ecpSlice("borneo-d0", "2606:ec80:10::20", true, nil)
	s.ResourceVersion = "2"
	if _, err := cs.DiscoveryV1().EndpointSlices("default").Update(context.Background(), s, metav1.UpdateOptions{}); err != nil {
		t.Fatal(err)
	}
	time.Sleep(300 * time.Millisecond)
	rec.mu.Lock()
	defer rec.mu.Unlock()
	if rec.upserts != 1 {
		t.Errorf("upserts = %d, want 1", rec.upserts)
	}
}
