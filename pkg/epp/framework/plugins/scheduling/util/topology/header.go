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
	"fmt"
	"strings"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	attrtopology "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/topology"
)

// ValidateHeaderName rejects a peer-topology header name that diverges from
// reqcommon.PeerTopologyHeaderKey. The coordinator's PrefillStep, decode
// proxy, and internal-header stripping all key off that single hardcoded
// name; a plugin configured with a different name would silently stop
// exchanging peer topology with the coordinator instead of failing at
// startup. Empty is allowed: it means "use the default" (topology-stamp
// handler) or "no header, single-EPP mode" (topology-affinity filter/scorer).
func ValidateHeaderName(name string) error {
	if name != "" && !strings.EqualFold(name, reqcommon.PeerTopologyHeaderKey) {
		return fmt.Errorf("peer topology header name %q is not supported; coordinator deployments require %q", name, reqcommon.PeerTopologyHeaderKey)
	}
	return nil
}

// Encode and Decode use a hand-rolled key=value wire format rather than JSON.
// JSON would be a smaller diff (struct tags plus Marshal/Unmarshal, with
// omitempty handling absent levels for free) at the cost of a noisier wire
// value and log line and a needed size cap. key=value costs a small
// hand-written parser and buys a self-describing value and
// order-independent, forward-compatible keys: an unknown key is ignored
// instead of rejected.

// Encode serializes a Topology as comma-separated key=value pairs, skipping
// empty fields. Encoding a nil Topology, or one with every field empty,
// returns "".
func Encode(t *attrtopology.Topology) string {
	if t == nil {
		return ""
	}
	fields := []struct {
		key, value string
	}{
		{"host", t.Hostname},
		{"rack", t.Rack},
		{"zone", t.Zone},
		{"region", t.Region},
	}
	var pairs []string
	for _, f := range fields {
		if f.value == "" || strings.ContainsAny(f.value, ",=") {
			continue
		}
		pairs = append(pairs, f.key+"="+f.value)
	}
	return strings.Join(pairs, ",")
}

// Decode parses a key=value wire value produced by Encode back into a
// Topology. Unknown keys are ignored, so a future level is additive.
// Malformed input (a pair with no "=") is skipped rather than rejected.
// Decode never returns an error; an empty or fully malformed value simply
// yields a Topology with every field empty.
func Decode(header string) *attrtopology.Topology {
	t := &attrtopology.Topology{}
	for _, pair := range strings.Split(header, ",") {
		key, value, ok := strings.Cut(pair, "=")
		if !ok {
			continue
		}
		switch key {
		case "host":
			t.Hostname = value
		case "rack":
			t.Rack = value
		case "zone":
			t.Zone = value
		case "region":
			t.Region = value
		}
	}
	return t
}
