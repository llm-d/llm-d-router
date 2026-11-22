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

package requestcontrol

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/sglanghttp"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
)

// TestDirector_ModelRewriteIfNeeded_SGLangGenerate covers the director half of
// #3357: TargetModelName is resolved from the model the parser read, but only
// along the model-rewrite path, which requires the parser to implement
// ModelNameRewriter. An SGLang /generate request carries a model and must reach
// the director with it; otherwise the precise prefix-cache scorer hashes the
// root block with an empty model and every block chained from the root misses.
func TestDirector_ModelRewriteIfNeeded_SGLangGenerate(t *testing.T) {
	parser := sglanghttp.NewSGLangHTTPParser()
	raw := []byte(`{"model":"llama-3-8b","input_ids":[1,2,3]}`)
	parsed, err := parser.ParseRequest(context.Background(), raw, map[string]string{":path": "/generate"})
	require.NoError(t, err)
	require.Equal(t, "llama-3-8b", parsed.Body.Model, "parser must read the /generate model")

	mockDs := &mockDatastore{}
	endpointCandidates := NewDatastoreEndpointCandidates(mockDs)
	director := NewDirectorWithConfig(mockDs, &mockScheduler{}, &mockAdmissionController{}, endpointCandidates, NewConfig(), "")

	// HandleRequest seeds IncomingModelName from the parsed body before resolving
	// the target.
	reqCtx := &handlers.RequestContext{
		Request:           &handlers.Request{RawBody: raw},
		Parser:            parser,
		IncomingModelName: parsed.Body.Model,
	}
	require.NoError(t, director.modelRewriteIfNeeded(context.Background(), reqCtx, parsed.Body))
	require.Equal(t, "llama-3-8b", reqCtx.TargetModelName,
		"target model must carry the model read from the /generate body")
}
