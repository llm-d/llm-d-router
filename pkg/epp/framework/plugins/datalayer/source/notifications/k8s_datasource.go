/*
Copyright 2026 The Kubernetes Authors.
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

package notifications

import (
	"context"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

var (
	_ fwkdl.DataSource         = (*K8sNotificationSource)(nil)
	_ fwkdl.NotificationSource = (*K8sNotificationSource)(nil)

	_ fwkdl.ScopedNotificationSource = (*K8sNotificationSource)(nil)
)

// K8sNotificationSource watches a single GVK and dispatches events to
// registered NotificationExtractors.
type K8sNotificationSource struct {
	typedName fwkplugin.TypedName
	gvk       schema.GroupVersionKind
	namespace string
	selector  labels.Selector
}

// K8sNotificationSourceOption configures a K8sNotificationSource.
type K8sNotificationSourceOption func(*K8sNotificationSource)

// WithNamespace restricts the watch to one namespace.
func WithNamespace(namespace string) K8sNotificationSourceOption {
	return func(s *K8sNotificationSource) { s.namespace = namespace }
}

// WithLabelSelector restricts the watch to objects matching selector.
func WithLabelSelector(selector labels.Selector) K8sNotificationSourceOption {
	return func(s *K8sNotificationSource) { s.selector = selector }
}

// NewK8sNotificationSource returns a new notification source for the given
// GVK. Without options it watches every object of the GVK.
func NewK8sNotificationSource(pluginType, pluginName string,
	gvk schema.GroupVersionKind, opts ...K8sNotificationSourceOption) *K8sNotificationSource {
	s := &K8sNotificationSource{
		typedName: fwkplugin.TypedName{Type: pluginType, Name: pluginName},
		gvk:       gvk,
	}
	for _, opt := range opts {
		opt(s)
	}
	return s
}

// TypedName returns the plugin type and name.
func (s *K8sNotificationSource) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// GVK returns the GroupVersionKind this source watches.
func (s *K8sNotificationSource) GVK() schema.GroupVersionKind {
	return s.gvk
}

// Namespace returns the watched namespace, or empty for all namespaces.
func (s *K8sNotificationSource) Namespace() string {
	return s.namespace
}

// LabelSelector returns the selector objects must match, or nil for all.
func (s *K8sNotificationSource) LabelSelector() labels.Selector {
	return s.selector
}

// Notify passes the event through for Runtime to dispatch to extractors.
// Returns nil to skip extractor dispatch.
func (s *K8sNotificationSource) Notify(_ context.Context, event fwkdl.NotificationEvent) (*fwkdl.NotificationEvent, error) {
	return &event, nil
}
