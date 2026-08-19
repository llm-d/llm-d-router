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

func TestParseLevel(t *testing.T) {
	valid := map[string]Level{"host": LevelHost, "rack": LevelRack, "zone": LevelZone, "region": LevelRegion}
	for s, want := range valid {
		got, err := ParseLevel(s)
		assert.NoError(t, err)
		assert.Equal(t, want, got)
	}

	_, err := ParseLevel("datacenter")
	assert.Error(t, err)
}

func TestCompare(t *testing.T) {
	full := &attrtopology.Topology{Hostname: "h1", Rack: "r1", Zone: "z1", Region: "reg1"}

	tests := []struct {
		name      string
		peer      *attrtopology.Topology
		candidate *attrtopology.Topology
		want      Level
	}{
		{"same host", full, full, LevelHost},
		{"same rack, different host", full, &attrtopology.Topology{Hostname: "h2", Rack: "r1", Zone: "z1", Region: "reg1"}, LevelRack},
		{"same zone, different rack", full, &attrtopology.Topology{Hostname: "h2", Rack: "r2", Zone: "z1", Region: "reg1"}, LevelZone},
		{"same region, different zone", full, &attrtopology.Topology{Hostname: "h2", Rack: "r2", Zone: "z2", Region: "reg1"}, LevelRegion},
		{"no common tier", full, &attrtopology.Topology{Hostname: "h2", Rack: "r2", Zone: "z2", Region: "reg2"}, LevelNone},
		{"empty vs empty never matches", &attrtopology.Topology{}, &attrtopology.Topology{}, LevelNone},
		{"missing field on candidate does not match", full, &attrtopology.Topology{Rack: "r1", Zone: "z1", Region: "reg1"}, LevelRack},
		{"candidate has only rack, matches at rack", full, &attrtopology.Topology{Rack: "r1"}, LevelRack},
		{"candidate has only zone, matches at zone", full, &attrtopology.Topology{Zone: "z1"}, LevelZone},
		{"nil peer", nil, full, LevelNone},
		{"nil candidate", full, nil, LevelNone},
		{"both nil", nil, nil, LevelNone},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, Compare(tt.peer, tt.candidate))
		})
	}
}
