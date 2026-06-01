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

// EndpointMetadata represents the relevant Kubernetes Pod state of an inference server.
type EndpointMetadata struct {
	NamespacedName types.NamespacedName
	PodName        string
	Address        string
	Port           string
	MetricsHost    string
	Labels         map[string]string
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
		NamespacedName: types.NamespacedName{
			Name:      epm.NamespacedName.Name,
			Namespace: epm.NamespacedName.Namespace,
		},
		PodName:     epm.PodName,
		Address:     epm.Address,
		Port:        epm.Port,
		MetricsHost: epm.MetricsHost,
		Labels:      clonedLabels,
		RankIndex:   epm.RankIndex,
	}
}

// GetRankIndex returns the rank index of this endpoint within the pool's
// TargetPorts list.
func (epm *EndpointMetadata) GetRankIndex() int {
	if epm == nil {
		return 0
	}
	return epm.RankIndex
}

// GetNamespacedName gets the namespace name of the Endpoint.
func (epm *EndpointMetadata) GetNamespacedName() types.NamespacedName {
	return epm.NamespacedName
}

// GetIPAddress returns the Endpoint's IP address.
func (epm *EndpointMetadata) GetIPAddress() string {
	return epm.Address
}

// GetPort returns the Endpoint's inference port.
func (epm *EndpointMetadata) GetPort() string {
	return epm.Port
}

// GetMetricsHost returns the Endpoint's metrics host (ip:port)
func (epm *EndpointMetadata) GetMetricsHost() string {
	return epm.MetricsHost
}
