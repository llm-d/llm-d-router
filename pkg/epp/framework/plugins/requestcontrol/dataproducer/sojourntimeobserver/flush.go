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

package sojourntimeobserver

import (
	"context"
	"errors"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrsojourn "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/sojourntime"
)

// Interval is the cadence at which the datalayer calls Dispatch.
func (p *Observer) Interval() time.Duration {
	return p.cfg.interval
}

// AppendExtractor rejects extractors: this dispatcher publishes its own state
// rather than sourcing data for others.
func (p *Observer) AppendExtractor(fwkplugin.Plugin) error {
	return errors.New("sojourn time observer does not accept extractors")
}

// Dispatch recomputes and publishes one endpoint's two attributes.
// The datalayer's collector calls it once per Interval. The digest snapshot is
// left as-is when either digest is below minSamples (so the scorer sees the
// endpoint as cold); the in-flight snapshot is refreshed unconditionally so
// consumers always see the newest in-flight list, capped in staleness by
// Interval.
func (p *Observer) Dispatch(ctx context.Context, ep fwkdl.Endpoint) error {
	if ep == nil || ep.GetMetadata() == nil {
		return nil
	}
	id := ep.GetMetadata().ID.String()
	state := p.stateForOrCreate(id)
	p.publish(ctx, id, state)
	p.publishInFlight(id, state)
	return nil
}

// publishInFlight snapshots the endpoint's in-flight index and swaps the
// pointer behind InFlightRequestsDataKey. Nil snapshot when the endpoint has
// no in-flight entries, so the DynamicAttribute closure returns nil and the
// endpoint reads as having no in-flight requests.
func (p *Observer) publishInFlight(id string, state *endpointState) {
	if state == nil {
		return
	}
	p.mu.RLock()
	entries := p.snapshotInFlightLocked(id)
	p.mu.RUnlock()
	if entries == nil {
		state.publishedInFlight.Store(nil)
		return
	}
	state.publishedInFlight.Store(&attrsojourn.InFlightRequestsSnapshot{Requests: entries})
}

// publish serializes both digests and swaps the snapshot pointer when both
// are warm. Neither digest is mutated here; AsBytes is a read-only path in
// caio/go-tdigest/v5, but the endpoint mutex is held anyway to prevent a
// concurrent Add from allocating centroids while AsBytes reads them.
func (p *Observer) publish(ctx context.Context, id string, state *endpointState) {
	if state == nil {
		return
	}
	state.mu.Lock()
	defer state.mu.Unlock()

	if state.ttft.Count() < p.cfg.minSamples || state.decode.Count() < p.cfg.minSamples {
		if debugLogger := log.FromContext(ctx).V(logutil.DEBUG); debugLogger.Enabled() {
			debugLogger.Info("sojourn snapshot skipped: cold",
				"endpoint", id,
				"ttftCount", state.ttft.Count(),
				"decodeCount", state.decode.Count(),
				"minSamples", p.cfg.minSamples)
		}
		return
	}

	ttftBytes, err := state.ttft.AsBytes()
	if err != nil {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(err, "sojourn TTFT digest serialize failed", "endpoint", id)
		return
	}
	decodeBytes, err := state.decode.AsBytes()
	if err != nil {
		log.FromContext(ctx).V(logutil.DEFAULT).Error(err, "sojourn decode digest serialize failed", "endpoint", id)
		return
	}
	snapshot := &attrsojourn.SojournEstimatorSnapshot{
		TtftDigest:   ttftBytes,
		DecodeDigest: decodeBytes,
	}
	state.published.Store(snapshot)

	if debugLogger := log.FromContext(ctx).V(logutil.DEBUG); debugLogger.Enabled() {
		debugLogger.Info("sojourn snapshot published",
			"endpoint", id,
			"ttftCount", state.ttft.Count(),
			"decodeCount", state.decode.Count(),
			"ttftBytes", len(ttftBytes),
			"decodeBytes", len(decodeBytes))
	}
}
