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

package tracing

import (
	"context"
	"net/http"

	"go.opentelemetry.io/otel/trace"
)

const (
	// FairnessIDHandoffHeader carries the fairness identity EPP resolved to the P/D sidecar.
	FairnessIDHandoffHeader = "x-llm-d-epp-fairness-id"
	// FairnessSourceHandoffHeader carries where EPP resolved that identity from.
	FairnessSourceHandoffHeader = "x-llm-d-epp-fairness-source"
)

// AttributionHandoff returns the attribution EPP hands to the P/D sidecar. It
// reports ok=false when the request's trace is not sampled, so the identity is not
// sent for traffic that no sidecar would trace.
func AttributionHandoff(ctx context.Context) (id, source string, ok bool) {
	if !trace.SpanContextFromContext(ctx).IsSampled() {
		return "", "", false
	}
	return RequestAttribution(ctx)
}

// WithAttributionHandoff wraps the sidecar's otelhttp handler. It starts request
// attribution from the EPP handoff, so the server span and its children carry the
// identity EPP resolved, and removes the handoff headers before the request is
// forwarded. An incomplete handoff or an unknown source resolves to the default
// sentinel. The headers are trusted because EPP replaces any client copy; a
// request that reaches the sidecar without passing EPP can still set them.
func WithAttributionHandoff(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		id := r.Header.Get(FairnessIDHandoffHeader)
		source := r.Header.Get(FairnessSourceHandoffHeader)
		r.Header.Del(FairnessIDHandoffHeader)
		r.Header.Del(FairnessSourceHandoffHeader)

		switch source {
		case AttributionSourceHeader, AttributionSourceAgentIdentity:
		default:
			id = ""
		}
		if id == "" {
			source = AttributionSourceDefault
		}

		ctx := BeginRequestAttribution(r.Context(), "")
		SetRequestAttribution(ctx, id, source)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}
