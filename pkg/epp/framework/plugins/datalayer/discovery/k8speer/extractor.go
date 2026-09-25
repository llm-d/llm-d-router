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

package k8speer

import (
	"context"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

var podGVK = schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}

type podExtractor struct {
	plugin *Plugin
}

var _ fwkdl.NotificationExtractor = (*podExtractor)(nil)

func (e *podExtractor) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: PluginType + "-pod-extractor", Name: e.plugin.typedName.Name + "/pod"}
}

func (e *podExtractor) GVK() schema.GroupVersionKind { return podGVK }

func (e *podExtractor) Extract(ctx context.Context, event fwkdl.NotificationEvent) error {
	if event.Object == nil {
		return nil
	}

	// A Pod's namespace never changes, so Pods outside it are never peers.
	// Skipping them keeps unrelated Pod churn off the handoff to Start.
	id := types.NamespacedName{Name: event.Object.GetName(), Namespace: event.Object.GetNamespace()}
	if id.Namespace != e.plugin.namespace {
		return nil
	}

	switch event.Type {
	case fwkdl.EventDelete:
		return e.plugin.emit(ctx, peerEvent{kind: fwkdl.EventDelete, id: id})
	case fwkdl.EventAddOrUpdate:
		return e.addOrUpdate(ctx, id, event.Object)
	default:
		return fmt.Errorf("%s: unhandled notification event type %v", PluginType, event.Type)
	}
}

// addOrUpdate upserts a Pod that matches the peer criteria and deletes any
// other Pod, so a peer that stops matching leaves the peer set.
func (e *podExtractor) addOrUpdate(ctx context.Context, id types.NamespacedName, obj *unstructured.Unstructured) error {
	pod := &corev1.Pod{}
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(obj.Object, pod); err != nil {
		return fmt.Errorf("convert Pod notification %s: %w", id, err)
	}

	if !e.plugin.acceptsPod(pod) {
		return e.plugin.emit(ctx, peerEvent{kind: fwkdl.EventDelete, id: id})
	}

	return e.plugin.emit(ctx, peerEvent{
		kind: fwkdl.EventAddOrUpdate,
		peer: &fwkdl.PeerMetadata{ID: id, Address: pod.Status.PodIP, Port: e.plugin.port},
	})
}
