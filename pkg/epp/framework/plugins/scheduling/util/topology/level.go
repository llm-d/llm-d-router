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

// Package topology provides comparison of endpoint Topology attributes,
// shared by the topology-affinity-filter and topology-affinity-scorer plugins.
package topology

import (
	"fmt"

	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
)

// Level names a topology tier, ordered tightest to loosest. Same host implies
// same rack, zone and region.
type Level int

const (
	LevelHost Level = iota
	LevelRack
	LevelZone
	LevelRegion
	// LevelNone means peer and candidate share no common tier.
	LevelNone
)

// levelNames maps each Level to its config-facing name, tightest to loosest.
var levelNames = map[Level]string{
	LevelHost:   "host",
	LevelRack:   "rack",
	LevelZone:   "zone",
	LevelRegion: "region",
}

// String returns the config-facing name of the level, or "none" for LevelNone
// and any other unrecognized value.
func (l Level) String() string {
	if s, ok := levelNames[l]; ok {
		return s
	}
	return "none"
}

// ParseLevel parses a config value into a Level. Valid values are "host",
// "rack", "zone", and "region".
func ParseLevel(s string) (Level, error) {
	for level, name := range levelNames {
		if name == s {
			return level, nil
		}
	}
	return LevelNone, fmt.Errorf("invalid topology level %q, must be one of host, rack, zone, region", s)
}

// Compare returns the tightest level at which peer and candidate share a
// non-empty, equal value, or LevelNone if they share none. An empty value
// never matches, including empty against empty, so an endpoint missing a
// field is never treated as co-located on that field. A nil peer or
// candidate always returns LevelNone.
func Compare(peer, candidate *attrtopology.Topology) Level {
	if peer == nil || candidate == nil {
		return LevelNone
	}
	switch {
	case matches(peer.Hostname, candidate.Hostname):
		return LevelHost
	case matches(peer.Rack, candidate.Rack):
		return LevelRack
	case matches(peer.Zone, candidate.Zone):
		return LevelZone
	case matches(peer.Region, candidate.Region):
		return LevelRegion
	default:
		return LevelNone
	}
}

func matches(a, b string) bool {
	return a != "" && a == b
}
