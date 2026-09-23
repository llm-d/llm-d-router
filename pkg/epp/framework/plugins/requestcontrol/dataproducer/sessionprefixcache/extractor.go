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

package sessionprefixcache

import (
	"context"

	"sigs.k8s.io/controller-runtime/pkg/log"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

var _ fwkdl.EndpointExtractor = &Producer{}

// Extract subscribes matching endpoints to per-pod KV events. Deleted
// endpoints and endpoints that stop matching lose their subscriber. Retiring
// the subscriber queues the endpoint's reset behind the events it already
// accepted, so the index is not cleared here: a direct clear could be
// overtaken by queued events and leave stale residency.
func (p *Producer) Extract(ctx context.Context, event fwkdl.EndpointEvent) error {
	meta := event.Endpoint.GetMetadata()
	if meta == nil || meta.ID.Name == "" {
		return nil
	}
	ctx = log.IntoContext(ctx, log.FromContext(ctx).WithName(p.typedName.String()))
	if event.Type == fwkdl.EventAddOrUpdate && p.subscriptions.Matches(meta.Labels) {
		return p.subscriptions.Ensure(ctx, meta.ID.String(), meta.Address, meta.Port, meta.GetRankIndex())
	}
	p.subscriptions.Remove(ctx, meta.ID.String())
	return nil
}
