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

package datalayer

import (
	"fmt"
	"maps"

	"k8s.io/apimachinery/pkg/types"
)

// ID uniquely identifies an endpoint. It aliases types.NamespacedName so the identity
// can later move off the Kubernetes-specific type without churning datastore keys.
// Every discovery source must set it uniquely.
type ID = types.NamespacedName

// EndpointMetadata describes an inference endpoint.
type EndpointMetadata struct {
	// ID is the endpoint's unique identity, and the datastore key.
	ID      ID
	Name    string
	Address string
	// NodeAddress is the node IP hosting this pod (pod.Status.HostIP).
	// Empty for non-Kubernetes discovery sources (e.g. file discovery).
	NodeAddress string
	Port        string
	MetricsHost string
	Labels      map[string]string
	// RankIndex is this endpoint's position in the pool's TargetPorts,
	// identifying the pod-local rank in multi-port deployments.
	RankIndex int
}

// String returns a string representation of the endpoint.
func (epm *EndpointMetadata) String() string {
	if epm == nil {
		return ""
	}
	return fmt.Sprintf("%+v", *epm)
}

// Clone returns a full copy of the object.
func (epm *EndpointMetadata) Clone() *EndpointMetadata {
	if epm == nil {
		return nil
	}

	clonedLabels := make(map[string]string, len(epm.Labels))
	maps.Copy(clonedLabels, epm.Labels)
	return &EndpointMetadata{
		ID: types.NamespacedName{
			Name:      epm.ID.Name,
			Namespace: epm.ID.Namespace,
		},
		Name:        epm.Name,
		Address:     epm.Address,
		NodeAddress: epm.NodeAddress,
		Port:        epm.Port,
		MetricsHost: epm.MetricsHost,
		Labels:      clonedLabels,
		RankIndex:   epm.RankIndex,
	}
}

// Equal reports whether two EndpointMetadata values describe the same endpoint
// metadata. A nil Labels map and an empty Labels map are treated as equal.
func (epm *EndpointMetadata) Equal(other *EndpointMetadata) bool {
	if epm == nil || other == nil {
		return epm == other
	}
	return epm.ID == other.ID &&
		epm.Name == other.Name &&
		epm.Address == other.Address &&
		epm.NodeAddress == other.NodeAddress &&
		epm.Port == other.Port &&
		epm.MetricsHost == other.MetricsHost &&
		epm.RankIndex == other.RankIndex &&
		maps.Equal(epm.Labels, other.Labels)
}

// GetRankIndex returns the rank index of this endpoint within the pool's
// TargetPorts list.
func (epm *EndpointMetadata) GetRankIndex() int {
	if epm == nil {
		return 0
	}
	return epm.RankIndex
}

// GetID returns the endpoint's unique identity.
func (epm *EndpointMetadata) GetID() ID {
	return epm.ID
}

// GetNamespacedName returns the endpoint's unique identity as a types.NamespacedName.
func (epm *EndpointMetadata) GetNamespacedName() types.NamespacedName {
	if epm == nil {
		return types.NamespacedName{}
	}
	return epm.ID
}

// GetIPAddress returns the Endpoint's IP address.
func (epm *EndpointMetadata) GetIPAddress() string {
	return epm.Address
}

// GetNodeAddress returns the IP of the node hosting this endpoint.
func (epm *EndpointMetadata) GetNodeAddress() string {
	if epm == nil {
		return ""
	}
	return epm.NodeAddress
}

// GetPort returns the Endpoint's inference port.
func (epm *EndpointMetadata) GetPort() string {
	return epm.Port
}

// GetMetricsHost returns the Endpoint's metrics host (ip:port)
func (epm *EndpointMetadata) GetMetricsHost() string {
	return epm.MetricsHost
}
