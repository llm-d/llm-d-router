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

// Package capture defines the records of a router profiler capture: one
// record per scheduling decision, written as one line of NDJSON and read by
// tools/doctor.
package capture

import "time"

// SchemaVersion is the Record version this package defines.
const SchemaVersion = 1

// Record is one scheduling decision. It holds no prompt text, headers or
// session keys: the prompt appears only as its prefix-cache block hashes.
type Record struct {
	SchemaVersion int       `json:"schemaVersion"`
	Time          time.Time `json:"time"`
	RequestID     string    `json:"requestID"`
	// Candidates are the endpoints the picker chose from.
	Candidates []string `json:"candidates"`
	// Endpoint is the endpoint the request was routed to, empty when
	// scheduling failed.
	Endpoint string `json:"endpoint,omitempty"`
	// HashScheme and BlockSize identify how BlockHashes were computed. Hashes
	// are comparable only between records that agree on both.
	HashScheme string `json:"hashScheme"`
	BlockSize  int    `json:"blockSize"`
	// BlockHashes are the prompt's chained block hashes in prompt order: a
	// hash names its block and every block before it.
	BlockHashes []uint64 `json:"blockHashes"`
	// PredictedCachedBlocks is the router's predicted prefix hit on Endpoint,
	// in blocks. Nil when no prefix producer ran.
	PredictedCachedBlocks *int `json:"predictedCachedBlocks,omitempty"`
	// EngineCachedTokens is the usage.prompt_tokens_details.cached_tokens the
	// engine returned. Nil when the response did not carry it.
	EngineCachedTokens *int `json:"engineCachedTokens,omitempty"`
}
