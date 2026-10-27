/*
Copyright 2025 The Kubernetes Authors.

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

package requestcontrol

import (
	"context"
	"net"
	"net/netip"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// --- DatastoreEndpointCandidates (The Delegate) ---

// EndpointCandidatesConfig holds configuration for the DatastoreEndpointCandidates.
type EndpointCandidatesConfig struct {
	DisableEndpointSubsetFilter bool
}

// EndpointCandidatesOption is a function that configures the EndpointCandidatesConfig.
type EndpointCandidatesOption func(*EndpointCandidatesConfig)

// WithDisableEndpointSubsetFilter sets the DisableEndpointSubsetFilter flag.
func WithDisableEndpointSubsetFilter(disable bool) EndpointCandidatesOption {
	return func(c *EndpointCandidatesConfig) {
		c.DisableEndpointSubsetFilter = disable
	}
}

// DatastoreEndpointCandidates implements contracts.EndpointCandidates by querying the EPP Datastore.
// It centralizes the logic for resolving endpoint candidates based on request metadata (specifically Envoy subset filters).
type DatastoreEndpointCandidates struct {
	datastore Datastore
	config    EndpointCandidatesConfig
}

var _ contracts.EndpointCandidates = &DatastoreEndpointCandidates{}

// NewDatastoreEndpointCandidates creates a new DatastoreEndpointCandidates.
func NewDatastoreEndpointCandidates(ds Datastore, opts ...EndpointCandidatesOption) *DatastoreEndpointCandidates {
	cfg := EndpointCandidatesConfig{
		DisableEndpointSubsetFilter: false,
	}
	for _, opt := range opts {
		opt(&cfg)
	}
	return &DatastoreEndpointCandidates{
		datastore: ds,
		config:    cfg,
	}
}

// Locate retrieves the list of endpoint candidates from the datastore that match the criteria defined in the request
// metadata.
//
// It supports:
// 1. Returning all endpoint candidates if no specific subset filter is present.
// 2. Returning a filtered list of endpoint candidates if "x-gateway-destination-endpoint-subset" is present.
func (d *DatastoreEndpointCandidates) Locate(ctx context.Context, requestMetadata map[string]any) []fwkdl.Endpoint {
	loggerTrace := log.FromContext(ctx).V(logutil.TRACE)

	// If the user explicitly disabled subset filtering, return the default pool (all endpoint candidates).
	if d.config.DisableEndpointSubsetFilter {
		loggerTrace.Info("endpoint subset filtering is explicitly disabled, returning all endpoint candidates")
		return d.datastore.PodList(datastore.AllPodsPredicate)
	}

	// Check if the subset filter namespace exists in metadata.
	// If not, we assume the request targets the default pool (all endpoint candidates).
	if requestMetadata == nil {
		return d.datastore.PodList(datastore.AllPodsPredicate)
	}

	subsetMap, found := requestMetadata[metadata.SubsetFilterNamespace].(map[string]any)
	if !found {
		return d.datastore.PodList(datastore.AllPodsPredicate)
	}

	// Check if the specific endpoint key exists within the subset map.
	endpointSubsetList, found := subsetMap[metadata.SubsetFilterKey].([]any)
	if !found {
		return d.datastore.PodList(datastore.AllPodsPredicate)
	}

	// If the filter key exists but the list is empty, it implies a filter that matched nothing upstream (or malformed
	// data), so we return nothing.
	if len(endpointSubsetList) == 0 {
		loggerTrace.Info("found empty subset filter in request metadata, returning empty endpoint candidate list")
		return []fwkdl.Endpoint{}
	}

	// Build a lookup map for efficient filtering.
	// The subset list contains strings in the format "<address>:<port>" (e.g., "10.0.1.0:8080").
	// We only care about the IP address for matching against PodMetrics.
	endpoints := sets.New[string]()
	for _, endpoint := range endpointSubsetList {
		epStr, ok := endpoint.(string)
		if !ok {
			loggerTrace.Info("ignoring non-string endpoint in subset list", "value", endpoint)
			continue
		}
		// Extract address from endpoint string.
		address := epStr
		ipAddress := epStr
		if host, _, err := net.SplitHostPort(epStr); err == nil {
			ipAddress = host
		} else if strings.HasPrefix(epStr, "[") && strings.HasSuffix(epStr, "]") {
			ipAddress = epStr[1 : len(epStr)-1]
		}
		if addr, err := netip.ParseAddr(ipAddress); err == nil {
			address = addr.String()
		} else if idx := strings.LastIndexByte(epStr, ':'); idx >= 0 {
			address = epStr[:idx]
		}
		endpoints.Insert(address)
	}

	// Query the Datastore with a predicate.
	podTotalCount := 0
	podFilteredList := d.datastore.PodList(func(pm fwkdl.Endpoint) bool {
		podTotalCount++
		// If the pod's IP is in our allowed map, include it.
		// Note: We use GetIPAddress() which should align with the subset address.
		if pod := pm.GetMetadata(); pod != nil {
			if _, found := endpoints[pod.GetIPAddress()]; found {
				return true
			}
		}
		return false
	})

	loggerTrace.Info("filtered endpoint candidates by subset filtering",
		"podTotalCount", podTotalCount,
		"filteredCount", len(podFilteredList))

	return podFilteredList
}
