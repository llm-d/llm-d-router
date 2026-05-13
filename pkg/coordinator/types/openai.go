package types

// ChatCompletionRequest represents an OpenAI-compatible chat completion request
// with extensions for disaggregated inference.
type ChatCompletionRequest struct {
	Model            string         `json:"model"`
	Messages         []Message      `json:"messages"`
	Stream           bool           `json:"stream,omitempty"`
	ECTransferParams map[string]any `json:"ec_transfer_params,omitempty"`
	KVTransferParams map[string]any `json:"kv_transfer_params,omitempty"`
}

type Message struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

type ImageURLContent struct {
	Type     string   `json:"type"`
	ImageURL ImageURL `json:"image_url"`
}

type ImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

// GenerateRequest is the request format for /inference/v1/generate (vLLM disaggregated protocol).
type GenerateRequest struct {
	RequestID        string              `json:"request_id,omitempty"`
	TokenIDs         []int               `json:"token_ids"`
	Features         *MultiModalFeatures `json:"features,omitempty"`
	SamplingParams   map[string]any      `json:"sampling_params,omitempty"`
	Model            string              `json:"model,omitempty"`
	Stream           bool                `json:"stream,omitempty"`
	KVTransferParams map[string]any      `json:"kv_transfer_params,omitempty"`
	ECTransferParams map[string]any      `json:"ec_transfer_params,omitempty"`
}

type MultiModalFeatures struct {
	MMHashes       map[string][]string          `json:"mm_hashes"`
	MMPlaceholders map[string][]PlaceholderInfo `json:"mm_placeholders"`
	KwargsData     map[string][]string          `json:"kwargs_data"`
}

type PlaceholderInfo struct {
	Offset int `json:"offset"`
	Length int `json:"length"`
}

// GenerateResponse is the response format from /inference/v1/generate.
type GenerateResponse struct {
	RequestID        string                   `json:"request_id"`
	Choices          []GenerateResponseChoice `json:"choices"`
	KVTransferParams map[string]any           `json:"kv_transfer_params,omitempty"`
	ECTransferParams map[string]any           `json:"ec_transfer_params,omitempty"`
}

type GenerateResponseChoice struct {
	Index        int    `json:"index"`
	FinishReason string `json:"finish_reason,omitempty"`
}
