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

// MembershipDisagreement describes each endpoint that differs between the two views.
type MembershipDisagreement map[types.NamespacedName]string

func (m MembershipDisagreement) Error() string {
	return fmt.Sprintf("%d routing endpoints differ from Kubernetes membership", len(m))
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
	mismatches := make(MembershipDisagreement)
	seen := make(map[types.NamespacedName]struct{})
	for _, ep := range ds.PodList(AllPodsPredicate) {
		meta := ep.GetMetadata()
		seen[meta.ID] = struct{}{}
		sampled := expected[meta.ID]
		switch {
		case sampled == nil:
			mismatches[meta.ID] = "absent from API sample"
		case !meta.Equal(sampled):
			mismatches[meta.ID] = "metadata differs"
		}
	}
	for id := range expected {
		if _, ok := seen[id]; !ok {
			mismatches[id] = "absent from datastore"
		}
	}
	if len(mismatches) > 0 {
		return expected, mismatches
	}
	return expected, nil
}
