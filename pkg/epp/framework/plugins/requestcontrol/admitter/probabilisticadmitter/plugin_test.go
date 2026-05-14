/*
Copyright 2026 The Kubernetes Authors.

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

package probabilisticadmitter

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"testing"

	errcommon "github.com/llm-d/llm-d-inference-scheduler/pkg/common/error"
	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	k8stypes "k8s.io/apimachinery/pkg/types"
)

func newEndpoint(name string, queueSize int, kvFraction float64) fwksched.Endpoint {
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: name}},
		&fwkdl.Metrics{
			WaitingQueueSize:    queueSize,
			KVCacheUsagePercent: kvFraction,
		},
		nil,
	)
}

func defaultParams() Parameters {
	return Parameters{
		QueueDepthThreshold:  5,
		KVCacheUtilThreshold: 0.8,
		Power:                5,
		K:                    300,
	}
}

func TestFactory(t *testing.T) {
	raw := json.RawMessage(`{"queueDepthThreshold":10,"kvCacheUtilThreshold":0.9,"power":3,"k":100}`)
	p, err := Factory("test-admitter", fwkplugin.StrictDecoder(raw), nil)
	if err != nil {
		t.Fatalf("Factory returned error: %v", err)
	}
	tn := p.TypedName()
	if tn.Type != Type {
		t.Errorf("expected type %q, got %q", Type, tn.Type)
	}
	if tn.Name != "test-admitter" {
		t.Errorf("expected name %q, got %q", "test-admitter", tn.Name)
	}
}

func TestFactory_Defaults(t *testing.T) {
	p, err := Factory("test", fwkplugin.StrictDecoder(nil), nil)
	if err != nil {
		t.Fatalf("Factory returned error: %v", err)
	}
	if p.TypedName().Type != Type {
		t.Errorf("expected type %q, got %q", Type, p.TypedName().Type)
	}
}

func TestFactory_InvalidJSON(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder(json.RawMessage(`{invalid}`)), nil)
	if err == nil {
		t.Fatal("expected error for invalid JSON")
	}
}

func TestFactory_ZeroQueueDepthThreshold(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder(json.RawMessage(`{"queueDepthThreshold":0}`)), nil)
	if err == nil {
		t.Fatal("expected error for queueDepthThreshold=0")
	}
}

func TestFactory_ZeroKVCacheUtilThreshold(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder(json.RawMessage(`{"kvCacheUtilThreshold":0}`)), nil)
	if err == nil {
		t.Fatal("expected error for kvCacheUtilThreshold=0")
	}
}

func TestFactory_NegativePower(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder(json.RawMessage(`{"power":-1}`)), nil)
	if err == nil {
		t.Fatal("expected error for negative power")
	}
}

func TestFactory_ZeroPower(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder(json.RawMessage(`{"power":0}`)), nil)
	if err == nil {
		t.Fatal("expected error for zero power")
	}
}

func TestFactory_NegativeK(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder(json.RawMessage(`{"k":-1}`)), nil)
	if err == nil {
		t.Fatal("expected error for negative k")
	}
}

func TestFactory_ZeroK(t *testing.T) {
	_, err := Factory("test", fwkplugin.StrictDecoder(json.RawMessage(`{"k":0}`)), nil)
	if err == nil {
		t.Fatal("expected error for zero k")
	}
}

func TestProtectedTiersAlwaysAdmitted(t *testing.T) {
	pods := []fwksched.Endpoint{newEndpoint("pod-0", 100, 0)}
	admitter := newAdmitter(defaultParams()).WithRandFn(func() float64 { return 0.0 })

	for _, priority := range []int{0, 100} {
		req := &fwksched.InferenceRequest{
			Objectives: fwksched.RequestObjectives{Priority: priority},
		}
		if err := admitter.AdmitRequest(context.Background(), req, pods); err != nil {
			t.Errorf("priority %d should always be admitted, got: %v", priority, err)
		}
	}
}

func TestDroppableRejectedAtHighSaturation(t *testing.T) {
	// QD=100, threshold=5 → saturation=20 → prob=1.0
	pods := []fwksched.Endpoint{newEndpoint("pod-0", 100, 0)}
	admitter := newAdmitter(defaultParams()).WithRandFn(func() float64 { return 0.0 })
	req := &fwksched.InferenceRequest{
		Objectives: fwksched.RequestObjectives{Priority: -1},
	}
	if err := admitter.AdmitRequest(context.Background(), req, pods); err == nil {
		t.Error("expected rejection at high saturation")
	}
}

func TestDroppableAdmittedAtZeroSaturation(t *testing.T) {
	pods := []fwksched.Endpoint{newEndpoint("pod-0", 0, 0)}
	admitter := newAdmitter(defaultParams()).WithRandFn(func() float64 { return 0.0 })
	req := &fwksched.InferenceRequest{
		Objectives: fwksched.RequestObjectives{Priority: -1},
	}
	if err := admitter.AdmitRequest(context.Background(), req, pods); err != nil {
		t.Errorf("expected admission at zero saturation, got: %v", err)
	}
}

func TestNilMetricsTreatedAsSaturated(t *testing.T) {
	pod := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "stale"}},
		nil,
		nil,
	)
	admitter := newAdmitter(defaultParams()).WithRandFn(func() float64 { return 0.0 })
	req := &fwksched.InferenceRequest{
		Objectives: fwksched.RequestObjectives{Priority: -1},
	}
	if err := admitter.AdmitRequest(context.Background(), req, []fwksched.Endpoint{pod}); err == nil {
		t.Error("expected rejection when pod has nil metrics")
	}
}

func TestNoPods(t *testing.T) {
	admitter := newAdmitter(defaultParams())
	req := &fwksched.InferenceRequest{
		Objectives: fwksched.RequestObjectives{Priority: -1},
	}
	if err := admitter.AdmitRequest(context.Background(), req, nil); err != nil {
		t.Errorf("no pods should admit, got: %v", err)
	}
}

func TestNilRequest(t *testing.T) {
	admitter := newAdmitter(defaultParams())
	pods := []fwksched.Endpoint{newEndpoint("pod-0", 0, 0)}
	if err := admitter.AdmitRequest(context.Background(), nil, pods); err != nil {
		t.Errorf("nil request should admit, got: %v", err)
	}
}

func TestMultiplePodsSaturationAveraging(t *testing.T) {
	// pod-a: max(10/5, 0/0.8)=2.0; pod-b: max(0/5, 0/0.8)=0.0 → avg=1.0 → prob=1.0
	pods := []fwksched.Endpoint{
		newEndpoint("pod-a", 10, 0),
		newEndpoint("pod-b", 0, 0),
	}
	admitter := newAdmitter(defaultParams()).WithRandFn(func() float64 { return 0.0 })
	req := &fwksched.InferenceRequest{
		Objectives: fwksched.RequestObjectives{Priority: -1},
	}
	if err := admitter.AdmitRequest(context.Background(), req, pods); err == nil {
		t.Error("expected rejection with avg saturation 1.0")
	}
}

func TestQuinticProperty(t *testing.T) {
	// sat=0.272/0.8=0.34 → sat^5*300≈1.36 → prob=1.0 → reject even at rand=0.999
	pods := []fwksched.Endpoint{newEndpoint("pod-0", 0, 0.272)}
	admitter := newAdmitter(defaultParams()).WithRandFn(func() float64 { return 0.999 })
	req := &fwksched.InferenceRequest{
		Objectives: fwksched.RequestObjectives{Priority: -1},
	}
	sat := 0.272 / 0.8
	if expected := math.Min(math.Pow(sat, 5)*300, 1.0); expected < 0.99 {
		t.Errorf("expected prob~1.0 at sat=%.4f, got %.4f", sat, expected)
	}
	if err := admitter.AdmitRequest(context.Background(), req, pods); err == nil {
		t.Error("expected rejection at sat≈0.34 (prob=1.0)")
	}
}

func TestErrorTypeIsStructured(t *testing.T) {
	// sat=1.0 → prob=1.0 → reject
	pods := []fwksched.Endpoint{newEndpoint("pod-0", 5, 0.8)}
	admitter := newAdmitter(defaultParams()).WithRandFn(func() float64 { return 0.0 })
	req := &fwksched.InferenceRequest{
		Objectives: fwksched.RequestObjectives{Priority: -1},
	}
	err := admitter.AdmitRequest(context.Background(), req, pods)
	if err == nil {
		t.Fatal("expected rejection")
	}
	var eppErr errcommon.Error
	if !errors.As(err, &eppErr) {
		t.Fatalf("expected errcommon.Error, got %T: %v", err, err)
	}
	if eppErr.Code != errcommon.ResourceExhausted {
		t.Errorf("expected code %q, got %q", errcommon.ResourceExhausted, eppErr.Code)
	}
}

func TestProbabilisticShedding(t *testing.T) {
	// sat=0.16/0.8=0.2 → prob=0.2^5*300≈0.096
	pods := []fwksched.Endpoint{newEndpoint("pod-0", 0, 0.16)}
	req := &fwksched.InferenceRequest{
		Objectives: fwksched.RequestObjectives{Priority: -1},
	}

	// rand=0.9 > prob≈0.096 → admit
	if err := newAdmitter(defaultParams()).WithRandFn(func() float64 { return 0.9 }).
		AdmitRequest(context.Background(), req, pods); err != nil {
		t.Errorf("expected admission with rand=0.9 > prob≈0.096, got: %v", err)
	}

	// rand=0.0 < prob≈0.096 → reject
	if err := newAdmitter(defaultParams()).WithRandFn(func() float64 { return 0.0 }).
		AdmitRequest(context.Background(), req, pods); err == nil {
		t.Error("expected rejection with rand=0.0 < prob≈0.096")
	}
}
