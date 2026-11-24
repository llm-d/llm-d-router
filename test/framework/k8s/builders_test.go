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

package k8s

import (
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPodWrapper(t *testing.T) {
	tests := []struct {
		name  string
		build func() *corev1.Pod
		check func(t *testing.T, pod *corev1.Pod)
	}{
		{
			name:  "MakePod sets only the name",
			build: func() *corev1.Pod { return MakePod("pod-1").ObjRef() },
			check: func(t *testing.T, pod *corev1.Pod) {
				if pod.Name != "pod-1" {
					t.Errorf("Name = %q, want %q", pod.Name, "pod-1")
				}
				if pod.Namespace != "" {
					t.Errorf("Namespace = %q, want empty", pod.Namespace)
				}
				if len(pod.Spec.Containers) != 0 {
					t.Errorf("Spec.Containers = %v, want empty", pod.Spec.Containers)
				}
				if len(pod.Status.Conditions) != 0 {
					t.Errorf("Status.Conditions = %v, want empty", pod.Status.Conditions)
				}
			},
		},
		{
			// Complete's default-namespace branch: reachable only when Namespace was
			// not called first.
			name:  "Complete defaults the namespace when unset",
			build: func() *corev1.Pod { return MakePod("pod-1").Complete().ObjRef() },
			check: func(t *testing.T, pod *corev1.Pod) {
				if pod.Namespace != "default" {
					t.Errorf("Namespace = %q, want %q", pod.Namespace, "default")
				}
			},
		},
		{
			name:  "Complete preserves an explicitly set namespace",
			build: func() *corev1.Pod { return MakePod("pod-1").Namespace("custom-ns").Complete().ObjRef() },
			check: func(t *testing.T, pod *corev1.Pod) {
				if pod.Namespace != "custom-ns" {
					t.Errorf("Namespace = %q, want %q", pod.Namespace, "custom-ns")
				}
			},
		},
		{
			name:  "Complete exposes the model server port",
			build: func() *corev1.Pod { return MakePod("pod-1").Complete().ObjRef() },
			check: func(t *testing.T, pod *corev1.Pod) {
				if len(pod.Spec.Containers) != 1 {
					t.Fatalf("Spec.Containers = %v, want exactly one container", pod.Spec.Containers)
				}
				c := pod.Spec.Containers[0]
				if c.Image == "" {
					t.Error("container Image is empty, want a default image")
				}
				if len(c.Ports) != 1 {
					t.Fatalf("container Ports = %v, want exactly one port", c.Ports)
				}
				if got := c.Ports[0].ContainerPort; got != DefaultTestPort {
					t.Errorf("ContainerPort = %d, want DefaultTestPort (%d)", got, DefaultTestPort)
				}
				if got := c.Ports[0].Protocol; got != corev1.ProtocolTCP {
					t.Errorf("Protocol = %q, want %q", got, corev1.ProtocolTCP)
				}
			},
		},
		{
			name: "Complete is idempotent across repeated calls",
			build: func() *corev1.Pod {
				return MakePod("pod-1").Complete().Complete().ObjRef()
			},
			check: func(t *testing.T, pod *corev1.Pod) {
				if len(pod.Spec.Containers) != 1 {
					t.Errorf("Spec.Containers = %v, want exactly one container", pod.Spec.Containers)
				}
				if pod.Namespace != "default" {
					t.Errorf("Namespace = %q, want %q", pod.Namespace, "default")
				}
			},
		},
		{
			name:  "Labels sets the pod labels",
			build: func() *corev1.Pod { return MakePod("pod-1").Labels(map[string]string{"app": "vllm"}).ObjRef() },
			check: func(t *testing.T, pod *corev1.Pod) {
				if got := pod.Labels["app"]; got != "vllm" {
					t.Errorf("Labels[app] = %q, want %q", got, "vllm")
				}
			},
		},
		{
			name:  "ReadyCondition marks the pod ready",
			build: func() *corev1.Pod { return MakePod("pod-1").ReadyCondition().ObjRef() },
			check: func(t *testing.T, pod *corev1.Pod) {
				if len(pod.Status.Conditions) != 1 {
					t.Fatalf("Status.Conditions = %v, want exactly one condition", pod.Status.Conditions)
				}
				cond := pod.Status.Conditions[0]
				if cond.Type != corev1.PodReady || cond.Status != corev1.ConditionTrue {
					t.Errorf("condition = %v/%v, want %v/%v", cond.Type, cond.Status, corev1.PodReady, corev1.ConditionTrue)
				}
			},
		},
		{
			name:  "IP sets the pod IP",
			build: func() *corev1.Pod { return MakePod("pod-1").IP("10.0.0.7").ObjRef() },
			check: func(t *testing.T, pod *corev1.Pod) {
				if pod.Status.PodIP != "10.0.0.7" {
					t.Errorf("Status.PodIP = %q, want %q", pod.Status.PodIP, "10.0.0.7")
				}
			},
		},
		{
			// A deletion timestamp without a finalizer is dropped by the apiserver, so
			// the builder must set both for the pod to read as terminating.
			name:  "DeletionTimestamp marks the pod terminating",
			build: func() *corev1.Pod { return MakePod("pod-1").DeletionTimestamp().ObjRef() },
			check: func(t *testing.T, pod *corev1.Pod) {
				if pod.DeletionTimestamp == nil {
					t.Fatal("DeletionTimestamp is nil, want a timestamp")
				}
				if pod.DeletionTimestamp.IsZero() {
					t.Error("DeletionTimestamp is the zero time, want a real time")
				}
				if len(pod.Finalizers) == 0 {
					t.Error("Finalizers is empty, want a finalizer so the timestamp is retained")
				}
			},
		},
		{
			name: "full fluent chain applies every setter",
			build: func() *corev1.Pod {
				return MakePod("pod-1").
					Namespace("ns-1").
					Labels(map[string]string{"app": "vllm", "role": "decode"}).
					Complete().
					ReadyCondition().
					IP("10.0.0.1").
					DeletionTimestamp().
					ObjRef()
			},
			check: func(t *testing.T, pod *corev1.Pod) {
				if pod.Name != "pod-1" || pod.Namespace != "ns-1" {
					t.Errorf("Name/Namespace = %q/%q, want %q/%q", pod.Name, pod.Namespace, "pod-1", "ns-1")
				}
				if len(pod.Labels) != 2 {
					t.Errorf("Labels = %v, want two entries", pod.Labels)
				}
				if len(pod.Spec.Containers) != 1 {
					t.Errorf("Spec.Containers = %v, want exactly one container", pod.Spec.Containers)
				}
				if len(pod.Status.Conditions) != 1 {
					t.Errorf("Status.Conditions = %v, want exactly one condition", pod.Status.Conditions)
				}
				if pod.Status.PodIP != "10.0.0.1" {
					t.Errorf("Status.PodIP = %q, want %q", pod.Status.PodIP, "10.0.0.1")
				}
				if pod.DeletionTimestamp == nil {
					t.Error("DeletionTimestamp is nil, want a timestamp")
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.check(t, tt.build())
		})
	}
}

func TestFromBase(t *testing.T) {
	const (
		baseNS = "base-ns"
		baseIP = "10.0.0.9"
	)

	base := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "base-pod",
			Namespace:   baseNS,
			Labels:      map[string]string{"app": "vllm"},
			Annotations: map[string]string{"note": "kept"},
		},
		Spec:   corev1.PodSpec{NodeName: "node-1"},
		Status: corev1.PodStatus{PodIP: baseIP},
	}

	tests := []struct {
		name  string
		build func() *corev1.Pod
		check func(t *testing.T, pod *corev1.Pod)
	}{
		{
			name:  "preserves the base pod fields",
			build: func() *corev1.Pod { return FromBase(base).ObjRef() },
			check: func(t *testing.T, pod *corev1.Pod) {
				if pod.Name != "base-pod" || pod.Namespace != baseNS {
					t.Errorf("Name/Namespace = %q/%q, want %q/%q", pod.Name, pod.Namespace, "base-pod", baseNS)
				}
				if pod.Annotations["note"] != "kept" {
					t.Errorf("Annotations[note] = %q, want %q", pod.Annotations["note"], "kept")
				}
				if pod.Spec.NodeName != "node-1" {
					t.Errorf("Spec.NodeName = %q, want %q", pod.Spec.NodeName, "node-1")
				}
				if pod.Status.PodIP != baseIP {
					t.Errorf("Status.PodIP = %q, want %q", pod.Status.PodIP, baseIP)
				}
			},
		},
		{
			// FromBase copies the Pod value, so scalar overrides on the wrapper must
			// not write back into the caller's base pod.
			name: "scalar overrides do not mutate the base pod",
			build: func() *corev1.Pod {
				return FromBase(base).Namespace("other-ns").IP("10.0.0.2").ObjRef()
			},
			check: func(t *testing.T, pod *corev1.Pod) {
				if pod.Namespace != "other-ns" || pod.Status.PodIP != "10.0.0.2" {
					t.Errorf("override = %q/%q, want %q/%q", pod.Namespace, pod.Status.PodIP, "other-ns", "10.0.0.2")
				}
				if base.Namespace != baseNS || base.Status.PodIP != baseIP {
					t.Errorf("base mutated to %q/%q, want %q/%q", base.Namespace, base.Status.PodIP, baseNS, baseIP)
				}
			},
		},
		{
			name:  "Complete keeps the base namespace",
			build: func() *corev1.Pod { return FromBase(base).Complete().ObjRef() },
			check: func(t *testing.T, pod *corev1.Pod) {
				if pod.Namespace != baseNS {
					t.Errorf("Namespace = %q, want %q", pod.Namespace, baseNS)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.check(t, tt.build())
		})
	}
}

// ObjRef must hand back the wrapper's own Pod, not a copy: callers pass the
// result to fake clients and expect later builder calls to be visible.
func TestObjRefAliasesTheWrappedPod(t *testing.T) {
	w := MakePod("pod-1")
	ref := w.ObjRef()

	if ref != &w.Pod {
		t.Fatalf("ObjRef() = %p, want &wrapper.Pod (%p)", ref, &w.Pod)
	}

	w.IP("10.0.0.5")
	if ref.Status.PodIP != "10.0.0.5" {
		t.Errorf("Status.PodIP via earlier ObjRef() = %q, want %q", ref.Status.PodIP, "10.0.0.5")
	}
}
