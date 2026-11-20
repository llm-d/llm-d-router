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

package endpointslice

import (
	"context"
	"fmt"

	discoveryv1 "k8s.io/api/discovery/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

var endpointSliceGVK = discoveryv1.SchemeGroupVersion.WithKind("EndpointSlice")

// sliceExtractor converts EndpointSlice notifications into slice events for
// Start.
type sliceExtractor struct {
	plugin *EndpointSliceDiscovery
}

var _ fwkdl.NotificationExtractor = (*sliceExtractor)(nil)

func (e *sliceExtractor) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: PluginType + "-slice-extractor", Name: e.plugin.typedName.Name + "/endpointslice"}
}

func (e *sliceExtractor) GVK() schema.GroupVersionKind { return endpointSliceGVK }

// Extract hands the slice change to Start and returns once it is applied.
// Slices outside the namespace or selector are ignored: the manager cache is
// scoped to them where the runner builds it, and the check keeps the plugin
// correct when the source is shared or unscoped.
func (e *sliceExtractor) Extract(ctx context.Context, event fwkdl.NotificationEvent) error {
	if event.Object == nil {
		return nil
	}
	key := types.NamespacedName{Namespace: event.Object.GetNamespace(), Name: event.Object.GetName()}
	if key.Namespace != e.plugin.namespace {
		return nil
	}

	switch event.Type {
	case fwkdl.EventDelete:
		return e.plugin.emit(ctx, sliceEvent{kind: fwkdl.EventDelete, key: key})
	case fwkdl.EventAddOrUpdate:
		// A slice that stops matching the selector leaves the set.
		if !e.plugin.selector.Matches(labels.Set(event.Object.GetLabels())) {
			return e.plugin.emit(ctx, sliceEvent{kind: fwkdl.EventDelete, key: key})
		}
		slice := &discoveryv1.EndpointSlice{}
		if err := runtime.DefaultUnstructuredConverter.FromUnstructured(event.Object.Object, slice); err != nil {
			return fmt.Errorf("%s: convert EndpointSlice %s: %w", PluginType, key, err)
		}
		return e.plugin.emit(ctx, sliceEvent{kind: fwkdl.EventAddOrUpdate, key: key, slice: slice})
	default:
		return fmt.Errorf("%s: unhandled notification event type %v", PluginType, event.Type)
	}
}
