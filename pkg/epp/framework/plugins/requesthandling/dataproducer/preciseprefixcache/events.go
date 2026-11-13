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

package preciseprefixcache

import (
	"context"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

// Only explicitly configured producers can bind to a shared event source.
type externalEventsProducer struct{ *Producer }

func (p *externalEventsProducer) ProcessEvents(ctx context.Context, source kvevents.EventSource, batch kvevents.EventBatch) error {
	return p.externalEvents.ProcessEvents(ctx, source, batch)
}

func (p *externalEventsProducer) Reset(ctx context.Context, source kvevents.EventSource) error {
	return p.externalEvents.Reset(ctx, source)
}
