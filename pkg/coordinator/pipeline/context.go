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

package pipeline

import (
	"net/http"
	"strings"
	"time"
)

var hopByHopHeaders = map[string]bool{
	"connection":          true,
	"keep-alive":          true,
	"proxy-authenticate":  true,
	"proxy-authorization": true,
	"te":                  true,
	"trailer":             true,
	"transfer-encoding":   true,
	"upgrade":             true,
}

var internalForwardingHeaders = map[string]bool{
	"epp-phase": true,
}

// ForwardedHeaders returns original request headers suitable for forwarding
// to upstream services, excluding hop-by-hop headers, Content-Length/Host, and
// coordinator-owned routing headers.
// Keys are normalized to lowercase so they do not collide by case with headers
// stamped explicitly by forwarding steps (e.g. x-request-id).
func (rc *RequestContext) ForwardedHeaders() map[string]string {
	out := make(map[string]string)
	if rc.OriginalHeaders == nil {
		return out
	}
	for key, vals := range rc.OriginalHeaders {
		lower := strings.ToLower(key)
		if hopByHopHeaders[lower] || internalForwardingHeaders[lower] || lower == "content-length" || lower == "host" || lower == "content-type" {
			continue
		}
		if len(vals) > 0 {
			out[lower] = vals[0]
		}
	}
	return out
}

// RequestContext carries all state for a single request through the pipeline.
type RequestContext struct {
	RequestID       string
	OriginalPath    string
	OriginalHeaders http.Header
	OriginalBody    []byte
	Body            map[string]any
	Model           string
	Stream          bool

	// ParseDuration is the time the server spent reading and JSON-parsing the
	// request body before the pipeline ran. Execute reports it as the first
	// entry in the step-timing summary.
	ParseDuration time.Duration

	TokenIDs          []int
	MultimodalEntries []MultimodalEntry
	// ECTransferParams is an ordered list (one entry per encode response).
	// Each entry is a single-key map: mm_hash -> opaque per-encoding transfer
	// descriptor (see the ec.Connector interface doc for the descriptor shape).
	// Populated by EncodeStep when the EC connector is ec-nixl; empty for
	// ec-shared-storage.
	ECTransferParams []map[string]any
	// KVTransferParams carries the prefill pod's KV-cache transfer hints to the
	// decode step. Populated by PrefillStep from the prefill response; consumed
	// by the KV connector when building the decode request.
	KVTransferParams map[string]any

	// ResponseWriter is used by decode steps to stream the final response to the client.
	ResponseWriter http.ResponseWriter
}

// MultimodalEntry describes one downloaded multimodal item (e.g. an image) and
// where it sits in the tokenized prompt. Index is its position in the request's
// multimodal list. Base64Data and ContentType come from the media download;
// Hash and KwargsData are filled in by the render step; Placeholder marks the
// span of placeholder tokens the encode step replaces.
type MultimodalEntry struct {
	Index       int
	Hash        string
	Base64Data  string
	ContentType string
	KwargsData  string
	Placeholder PlaceholderRange
}

// PlaceholderRange is the span of placeholder tokens for one multimodal entry
// in the tokenized prompt: Offset is the index of the first placeholder token
// and Length is the number of placeholder tokens.
type PlaceholderRange struct {
	Offset int `json:"offset"`
	Length int `json:"length"`
}
