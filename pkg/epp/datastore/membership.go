/*
Copyright 2026 The llm-d Authors.

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

package datastore

import (
	"errors"
	"fmt"
	"slices"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	podutil "github.com/llm-d/llm-d-router/pkg/epp/util/pod"
)

// VerifyPool checks routing configuration independently of Pod discovery availability.
func VerifyPool(ds Datastore, pool *datalayer.EndpointPool) error {
	current, err := ds.PoolGet()
	if err != nil {
		return err
	}
	if current.Namespace != pool.Namespace || current.Name != pool.Name ||
		current.AppProtocol != pool.AppProtocol || !selectorEqual(current.Selector, pool.Selector) ||
		!slices.Equal(current.TargetPorts, pool.TargetPorts) {
		return errors.New("routing pool does not match Kubernetes membership")
	}
	return nil
}

// VerifyMembership returns authoritative metadata and reports disagreement with routing membership.
func VerifyMembership(ds Datastore, pool *datalayer.EndpointPool, pods []corev1.Pod) (map[types.NamespacedName]*fwkdl.EndpointMetadata, error) {
	if err := VerifyPool(ds, pool); err != nil {
		return nil, err
	}
	expected := make(map[types.NamespacedName]*fwkdl.EndpointMetadata)
	for i := range pods {
		pod := &pods[i]
		if pod.Namespace != pool.Namespace || !podutil.IsPodReady(pod) || !pool.Selector.Matches(labels.Set(pod.Labels)) {
			continue
		}
		for _, meta := range endpointMetadataForPod(pod, pool) {
			expected[meta.ID] = meta
		}
	}
	endpoints := ds.PodList(AllPodsPredicate)
	if len(endpoints) != len(expected) {
		return expected, errors.New("routing endpoints do not match Kubernetes membership")
	}
	for _, ep := range endpoints {
		meta := ep.GetMetadata()
		if !meta.Equal(expected[meta.ID]) {
			return expected, fmt.Errorf("routing endpoint %s does not match Kubernetes membership", meta.ID)
		}
	}
	return expected, nil
}
