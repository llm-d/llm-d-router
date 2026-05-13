package pipeline

import (
	"net/http"
	"time"
)

// RequestContext carries all state for a single request through the pipeline.
type RequestContext struct {
	RequestID    string
	OriginalPath string
	OriginalBody []byte
	Body         map[string]any
	Model        string
	Stream       bool

	TokenIDs          []int
	MultimodalEntries []MultimodalEntry
	ECTransferParams  []map[string]any
	KVTransferParams  map[string]any

	ResponseWriter http.ResponseWriter
	Flusher        http.Flusher

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
