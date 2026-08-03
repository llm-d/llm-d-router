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

package file

import (
	"context"
	"encoding/json"
	"fmt"
	"net"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// MultiClusterPluginType discovers peer clusters as endpoints for a cluster-scoped EPP.
const MultiClusterPluginType = "multicluster-file-discovery"

// MultiClusterFactory builds a file discovery whose endpoints are peer clusters. A
// cluster is reached by its gateway host, so the address may be a hostname or an IP,
// unlike the pod file discovery which requires an IPv4 pod address.
func MultiClusterFactory(name string, parameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	fd, err := newFileDiscovery(MultiClusterPluginType, name, parameters, validateHostOrIP)
	if err != nil {
		return nil, err
	}
	return &multiClusterFileDiscovery{FileDiscovery: fd}, nil
}

// multiClusterFileDiscovery adds cluster-endpoint behavior the stock file discovery does not
// carry, so the pod discovery never interprets cluster-only labels.
type multiClusterFileDiscovery struct {
	*FileDiscovery
}

var _ fwkdl.EndpointDiscovery = (*multiClusterFileDiscovery)(nil)

// Start wraps the notifier so each endpoint's metrics host is resolved from its labels before
// it reaches the datastore, on the initial load and every reload.
func (m *multiClusterFileDiscovery) Start(ctx context.Context, notifier fwkdl.DiscoveryNotifier) error {
	return m.FileDiscovery.Start(ctx, metricsHostNotifier{notifier})
}

// metricsHostNotifier resolves the metrics host from labels on each upsert.
type metricsHostNotifier struct {
	fwkdl.DiscoveryNotifier
}

func (n metricsHostNotifier) Upsert(meta *fwkdl.EndpointMetadata) {
	metricsHostFromLabels(meta)
	n.DiscoveryNotifier.Upsert(meta)
}

// metricsHostFromLabels lets a peer publish its metrics endpoint apart from its routable
// address, via a metricsAddress label (and an optional metricsPort label, defaulting to the
// endpoint port). A peer whose EPP serves pool metrics on a different host or port than its
// gateway needs this. Without the label the metrics host stays the endpoint's own address.
func metricsHostFromLabels(meta *fwkdl.EndpointMetadata) {
	addr := meta.Labels["metricsAddress"]
	if addr == "" {
		return
	}
	port := meta.Labels["metricsPort"]
	if port == "" {
		port = meta.Port
	}
	meta.MetricsHost = net.JoinHostPort(addr, port)
}

// validateHostOrIP accepts an IP literal (v4 or v6) or an RFC-1123 DNS hostname, since a
// cluster gateway may be reached by either.
func validateHostOrIP(address string) error {
	if net.ParseIP(address) != nil {
		return nil
	}
	if isValidHostname(address) {
		return nil
	}
	return fmt.Errorf("invalid host %q", address)
}

// isValidHostname reports whether h is an RFC-1123 hostname: dot-separated LDH labels
// (letters, digits, hyphen) of 1-63 chars, no label starting or ending with a hyphen, an
// optional trailing dot, and at least one letter so a numeric string cannot pass as a
// host. It is a character scan rather than a regexp, mirroring net.isDomainName.
func isValidHostname(h string) bool {
	l := len(h)
	if l == 0 || l > 254 || (l == 254 && h[l-1] != '.') {
		return false
	}
	last := byte('.')
	hasLetter := false
	partLen := 0
	for i := 0; i < len(h); i++ {
		c := h[i]
		switch {
		default:
			return false
		case 'a' <= c && c <= 'z' || 'A' <= c && c <= 'Z':
			hasLetter = true
			partLen++
		case '0' <= c && c <= '9':
			partLen++
		case c == '-':
			if last == '.' { // label cannot start with a hyphen
				return false
			}
			partLen++
		case c == '.':
			if last == '.' || last == '-' { // empty label or trailing hyphen
				return false
			}
			if partLen > 63 {
				return false
			}
			partLen = 0
		}
		last = c
	}
	if last == '-' || partLen > 63 {
		return false
	}
	return hasLetter
}
