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

package topology

import (
	"testing"

	"github.com/stretchr/testify/assert"

	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
)

func TestEncodeDecode_RoundTrip(t *testing.T) {
	cases := []struct {
		name string
		topo *attrtopology.Topology
	}{
		{"full", &attrtopology.Topology{Hostname: "node12", Rack: "r7", Zone: "us-east1-a", Region: "us-east1"}},
		{"partial-no-rack", &attrtopology.Topology{Hostname: "node12", Zone: "us-east1-a"}},
		{"host-only", &attrtopology.Topology{Hostname: "node12"}},
		{"empty", &attrtopology.Topology{}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := Decode(Encode(tc.topo))
			assert.Equal(t, tc.topo, got)
		})
	}
}

func TestEncode_NilReturnsEmpty(t *testing.T) {
	assert.Equal(t, "", Encode(nil))
}

func TestEncode_SkipsValueWithSeparator(t *testing.T) {
	got := Encode(&attrtopology.Topology{Hostname: "node12", Rack: "r7,r8", Zone: "a=b"})
	assert.Equal(t, "host=node12", got)
}

func TestDecode_UnknownKeyIgnored(t *testing.T) {
	got := Decode("host=node12,gpu=h100")
	assert.Equal(t, &attrtopology.Topology{Hostname: "node12"}, got)
}

func TestDecode_MalformedInputYieldsEmptyTopology(t *testing.T) {
	got := Decode("not-a-pair,,host")
	assert.Equal(t, &attrtopology.Topology{}, got)
}

func TestDecode_EmptyString(t *testing.T) {
	got := Decode("")
	assert.Equal(t, &attrtopology.Topology{}, got)
}
