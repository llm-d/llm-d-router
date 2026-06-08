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

package session_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
)

// stubEndpoint is a minimal scheduling.Endpoint used to exercise
// EndpointBoundForm against shapes the real NewEndpoint constructor
// rejects (e.g. nil metadata, which NewEndpoint would dereference).
type stubEndpoint struct {
	meta *fwkdl.EndpointMetadata
}

func (s *stubEndpoint) GetMetadata() *fwkdl.EndpointMetadata { return s.meta }
func (s *stubEndpoint) GetMetrics() *fwkdl.Metrics           { return nil }
func (s *stubEndpoint) String() string                       { return "" }
func (s *stubEndpoint) Get(string) (fwkdl.Cloneable, bool)   { return nil, false }
func (s *stubEndpoint) Put(string, fwkdl.Cloneable)          {}
func (s *stubEndpoint) Keys() []string                       { return nil }
func (s *stubEndpoint) Clone() fwkdl.AttributeMap            { return nil }

func TestEndpointBoundForm(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		endpoint fwksched.Endpoint
		want     attrsession.BoundEndpoint
	}{
		{
			name:     "nil endpoint",
			endpoint: nil,
			want:     "",
		},
		{
			name:     "nil metadata",
			endpoint: &stubEndpoint{meta: nil},
			want:     "",
		},
		{
			name: "empty address",
			endpoint: &stubEndpoint{meta: &fwkdl.EndpointMetadata{
				NamespacedName: k8stypes.NamespacedName{Name: "pod-1", Namespace: "default"},
				Address:        "",
				Port:           "8080",
			}},
			want: "",
		},
		{
			name: "empty port",
			endpoint: &stubEndpoint{meta: &fwkdl.EndpointMetadata{
				NamespacedName: k8stypes.NamespacedName{Name: "pod-1", Namespace: "default"},
				Address:        "10.0.0.1",
				Port:           "",
			}},
			want: "",
		},
		{
			name: "ipv4 happy path",
			endpoint: &stubEndpoint{meta: &fwkdl.EndpointMetadata{
				NamespacedName: k8stypes.NamespacedName{Name: "pod-1", Namespace: "default"},
				Address:        "10.0.0.1",
				Port:           "8080",
			}},
			want: "10.0.0.1:8080",
		},
		{
			name: "ipv6 host wrapped in brackets",
			endpoint: &stubEndpoint{meta: &fwkdl.EndpointMetadata{
				NamespacedName: k8stypes.NamespacedName{Name: "pod-1", Namespace: "default"},
				Address:        "::1",
				Port:           "8080",
			}},
			want: "[::1]:8080",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			got := attrsession.EndpointBoundForm(tc.endpoint)
			assert.Equal(t, tc.want, got)
		})
	}
}

// TestEndpointBoundForm_RealEndpoint pins the helper against an Endpoint
// built by the framework's own constructor, so the test catches the
// case where the helper drifts from the real Endpoint shape.
func TestEndpointBoundForm_RealEndpoint(t *testing.T) {
	t.Parallel()

	ep := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Name: "pod-1", Namespace: "default"},
			Address:        "10.0.0.1",
			Port:           "8080",
		},
		&fwkdl.Metrics{},
		nil,
	)
	assert.Equal(t, attrsession.BoundEndpoint("10.0.0.1:8080"), attrsession.EndpointBoundForm(ep))
}
