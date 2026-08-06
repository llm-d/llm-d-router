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
	"k8s.io/apimachinery/pkg/util/validation"
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

// validateHostOrIP accepts an IP literal (v4 or v6) or a DNS-1123 subdomain, since a
// cluster gateway may be reached by either.
func validateHostOrIP(address string) error {
	if net.ParseIP(address) != nil {
		return nil
	}
	if len(validation.IsDNS1123Subdomain(address)) == 0 {
		return nil
	}
	return fmt.Errorf("invalid host %q", address)
}
