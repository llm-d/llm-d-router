/*
Copyright 2026 The Kubernetes Authors.

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

package zmqmetrics

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/vmihailenco/msgpack/v5"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

func TestZMQExtractor_UnsupportedEngine(t *testing.T) {
	_, err := NewZMQMetricsExtractor("test-extractor", "no-such-engine")
	assert.Error(t, err)
}

func TestZMQExtractor_MissingEngine(t *testing.T) {
	_, err := NewZMQMetricsExtractor("test-extractor", "")
	assert.Error(t, err)
}

func TestZMQExtractorFactory_EngineParameter(t *testing.T) {
	t.Run("no parameters is an error", func(t *testing.T) {
		_, err := ZMQExtractorFactory("ext", nil, nil)
		assert.Error(t, err)
	})

	t.Run("empty engine is an error", func(t *testing.T) {
		params := json.NewDecoder(strings.NewReader(`{}`))
		_, err := ZMQExtractorFactory("ext", params, nil)
		assert.Error(t, err)
	})

	t.Run("sglang engine", func(t *testing.T) {
		params := json.NewDecoder(strings.NewReader(`{"engine": "sglang"}`))
		p, err := ZMQExtractorFactory("ext", params, nil)
		require.NoError(t, err)

		ext, ok := p.(*Extractor)
		require.True(t, ok)

		payload, err := msgpack.Marshal([]any{"LoadStat", 1, 2, 100, 400, 0})
		require.NoError(t, err)

		ep := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{Address: "10.0.0.1"}, nil)
		require.NoError(t, ext.Extract(context.Background(), fwkdl.StreamInput[[]byte]{Payload: payload, Endpoint: ep}))
		assert.Equal(t, 1, ep.GetMetrics().RunningRequestsSize)
	})

	t.Run("unsupported engine", func(t *testing.T) {
		params := json.NewDecoder(strings.NewReader(`{"engine": "bogus"}`))
		_, err := ZMQExtractorFactory("ext", params, nil)
		assert.Error(t, err)
	})
}
