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
	"trailers":            true,
	"transfer-encoding":   true,
	"upgrade":             true,
}

// ForwardedHeaders returns original request headers suitable for forwarding
// to upstream services, excluding hop-by-hop headers and Content-Length/Host.
func (rc *RequestContext) ForwardedHeaders() map[string]string {
	out := make(map[string]string)
	if rc.OriginalHeaders == nil {
		return out
	}
	for key, vals := range rc.OriginalHeaders {
		lower := strings.ToLower(key)
		if hopByHopHeaders[lower] || lower == "content-length" || lower == "host" || lower == "content-type" {
			continue
		}
		if len(vals) > 0 {
			out[key] = vals[0]
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

	TokenIDs          []int
	MultimodalEntries []MultimodalEntry
	// ECTransferParams is an ordered list (one entry per encode response).
	// Each entry is a single-key map: mm_hash -> opaque per-encoding transfer
	// descriptor (see the ec.Connector interface doc for the descriptor shape).
	// Populated by EncodeStep when the EC connector is ec-nixl; empty for
	// ec-shared-storage.
	ECTransferParams []map[string]any
	KVTransferParams map[string]any

	// ResponseWriter is used by decode steps to stream the final response to the client.
	ResponseWriter http.ResponseWriter

	StartTime time.Time
}

type MultimodalEntry struct {
	Index       int
	Hash        string
	Base64Data  string
	ContentType string
	KwargsData  string
	Placeholder PlaceholderRange
}

type PlaceholderRange struct {
	Offset int `json:"offset"`
	Length int `json:"length"`
}
