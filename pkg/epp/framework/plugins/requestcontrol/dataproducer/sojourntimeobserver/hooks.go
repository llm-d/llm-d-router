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
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

var (
	_ requestcontrol.PreRequest            = &Observer{}
	_ requestcontrol.ResponseBodyProcessor = &Observer{}
)

// PreRequest records the dispatch timestamp for the primary target endpoint
// in the fleet-wide in-flight index. Always returns nil: failing to record
// an observation is never a reason to reject a request.
func (p *Observer) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, result *fwksched.SchedulingResult) error {
	endpoint := primaryTarget(result)
	if request == nil || request.RequestID == "" || endpoint == nil {
		log.FromContext(ctx).V(logutil.DEBUG).Info("Skipping sojourn tracking: no request ID or no primary target")
		return nil
	}
	endpointID := endpoint.GetMetadata().ID.String()
	p.noteDispatch(endpointID, request.RequestID, time.Now())
	return nil
}

// primaryTarget returns the endpoint the primary profile selected, or nil.
// Sojourn belongs to whichever endpoint served the request.
func primaryTarget(result *fwksched.SchedulingResult) fwksched.Endpoint {
	if result == nil {
		return nil
	}
	primary := result.ProfileResults[result.PrimaryProfileName]
	if primary == nil || len(primary.TargetEndpoints) == 0 {
		return nil
	}
	if endpoint := primary.TargetEndpoints[0]; endpoint != nil && endpoint.GetMetadata() != nil {
		return endpoint
	}
	return nil
}

// ResponseBody handles the three chunk shapes:
//
//   - StartOfStream && !EndOfStream: streaming first chunk. Emit one TTFT
//     sample and record firstChunkAt on the in-flight entry.
//   - StartOfStream && EndOfStream:  single-chunk (non-streaming) response.
//     Fire onFirstChunk then onEndOfStream in order on the same event.
//     TTFT is firstChunkAt - dispatchedAt; decode is 0. The decode digest
//     correctly records this as a distributional feature.
//   - !StartOfStream && EndOfStream: streaming terminal chunk. Emit one
//     decode sample and clear the in-flight entry.
//   - !StartOfStream && !EndOfStream: interior streaming chunk. Ignored.
func (p *Observer) ResponseBody(ctx context.Context, request *fwksched.InferenceRequest,
	response *requestcontrol.Response, _ *fwkdl.EndpointMetadata) {
	if request == nil || response == nil || request.RequestID == "" {
		return
	}

	if response.StartOfStream {
		now := time.Now()
		p.onFirstChunk(ctx, request.RequestID, now)
		if response.EndOfStream {
			p.onEndOfStream(ctx, request.RequestID, now)
		}
		return
	}

	if response.EndOfStream {
		p.onEndOfStream(ctx, request.RequestID, time.Now())
	}
}

// onFirstChunk emits one TTFT sample and stamps firstChunkAt on the in-flight
// entry.
func (p *Observer) onFirstChunk(ctx context.Context, requestID string, firstChunkAt time.Time) {
	dispatchedAt, endpointID := p.noteFirstChunk(requestID, firstChunkAt)
	if endpointID == "" {
		// No in-flight entry — PreRequest was skipped or the endpoint was
		// deleted while the request was still in flight.
		return
	}
	ttft := firstChunkAt.Sub(dispatchedAt).Seconds()
	if ttft < 0 {
		ttft = 0
	}
	p.addTTFT(endpointID, ttft)

	if debugLogger := log.FromContext(ctx).V(logutil.DEBUG); debugLogger.Enabled() {
		debugLogger.Info("sojourn-ttft-observation",
			"requestID", requestID, "endpoint", endpointID, "ttftSeconds", ttft)
	}
}

// onEndOfStream emits one decode sample and clears the in-flight entry.
func (p *Observer) onEndOfStream(ctx context.Context, requestID string, endOfStreamAt time.Time) {
	firstChunkAt, endpointID := p.noteEndOfStream(requestID)
	if endpointID == "" {
		// No in-flight entry — a terminal chunk without a prior start, or
		// the endpoint was deleted mid-flight.
		return
	}
	if firstChunkAt.IsZero() {
		// EndOfStream arrived without any StartOfStream. The observer has
		// no TTFT sample and cannot compute a decode sample either; drop
		// silently. This is unusual — the streaming framing should
		// guarantee a StartOfStream chunk — but a defensive check avoids
		// a nonsensical decode = endOfStreamAt - zero.
		return
	}
	decode := endOfStreamAt.Sub(firstChunkAt).Seconds()
	if decode < 0 {
		decode = 0
	}
	p.addDecode(endpointID, decode)

	if debugLogger := log.FromContext(ctx).V(logutil.DEBUG); debugLogger.Enabled() {
		debugLogger.Info("sojourn-decode-observation",
			"requestID", requestID, "endpoint", endpointID, "decodeSeconds", decode)
	}
}
