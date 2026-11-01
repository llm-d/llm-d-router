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

package vllmgrpc

import (
	"context"
	"testing"
)

// The handler calls ParseResponse once more with an empty body when a gRPC
// stream ends on its trailer. That must report no usage, not a parse error that
// would be logged on every completed streaming request.
func TestParseResponse_EmptyBodyReturnsNoUsage(t *testing.T) {
	p := NewVllmGRPCParser()
	for _, body := range [][]byte{nil, {}} {
		got, err := p.ParseResponse(context.Background(), body, map[string]string{}, true)
		if err != nil {
			t.Fatalf("empty body: unexpected error: %v", err)
		}
		if got == nil || got.Usage != nil {
			t.Fatalf("empty body: want a parsed response carrying no usage, got %+v", got)
		}
	}
}
