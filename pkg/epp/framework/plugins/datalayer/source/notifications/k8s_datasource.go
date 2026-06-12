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

package notifications

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime/schema"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

var (
	_ fwkdl.DataSource         = (*K8sNotificationSource)(nil)
	_ fwkdl.NotificationSource = (*K8sNotificationSource)(nil)
)

// K8sNotificationSource watches a single GVK and dispatches events to
// registered NotificationExtractors.
type K8sNotificationSource struct {
	typedName fwkplugin.TypedName
	gvk       schema.GroupVersionKind
}

// NewK8sNotificationSource returns a new notification source for the given GVK.
func NewK8sNotificationSource(pluginType, pluginName string,
	gvk schema.GroupVersionKind) *K8sNotificationSource {
	return &K8sNotificationSource{
		typedName: fwkplugin.TypedName{Type: pluginType, Name: pluginName},
		gvk:       gvk,
	}
}

// TypedName returns the plugin type and name.
func (s *K8sNotificationSource) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// GVK returns the GroupVersionKind this source watches.
func (s *K8sNotificationSource) GVK() schema.GroupVersionKind {
	return s.gvk
}

// Notify passes the event through for Runtime to dispatch to extractors.
// Returns nil to skip extractor dispatch.
func (s *K8sNotificationSource) Notify(_ context.Context, event fwkdl.NotificationEvent) (*fwkdl.NotificationEvent, error) {
	return &event, nil
}
