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

package engineadapter //nolint:testpackage // tests reset the unexported registry

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

// fakeAdapter stands in for a third engine's adapter with its own topic
// scheme: "<pod>|<model>". It decodes no payload.
type fakeAdapter struct{}

func (fakeAdapter) ParseMessage(msg *kvevents.RawMessage) (string, string, kvevents.EventBatch, error) {
	podID, modelName, _ := strings.Cut(msg.Topic, "|")
	return podID, modelName, kvevents.EventBatch{}, nil
}

func (fakeAdapter) ShardingKey(msg *kvevents.RawMessage) string {
	podID, _, _ := strings.Cut(msg.Topic, "|")
	return podID
}

// registerForTest registers factory under engineType and restores the
// previous registration when the test ends.
func registerForTest(t *testing.T, engineType string, factory Factory) {
	t.Helper()
	previous, registered := registry[engineType]
	Register(engineType, factory)
	t.Cleanup(func() {
		if registered {
			registry[engineType] = previous
		} else {
			delete(registry, engineType)
		}
	})
}

func TestNewAdapter_BuiltInEngines(t *testing.T) {
	for engineType, want := range map[string]kvevents.EngineAdapter{
		"":               &VLLMAdapter{},
		EngineTypeVLLM:   &VLLMAdapter{},
		EngineTypeSGLang: &SGLangAdapter{},
	} {
		got, err := NewAdapter(engineType)
		require.NoError(t, err, "engine type %q", engineType)
		assert.IsType(t, want, got, "engine type %q", engineType)
	}
	assert.Equal(t, []string{EngineTypeSGLang, EngineTypeVLLM}, engineTypes())
}

func TestNewAdapter_RegisteredEngine(t *testing.T) {
	built := 0
	registerForTest(t, "own-format", func() kvevents.EngineAdapter {
		built++
		return &fakeAdapter{}
	})

	first, err := NewAdapter("own-format")
	require.NoError(t, err)
	_, err = NewAdapter("own-format")
	require.NoError(t, err)
	assert.Equal(t, 2, built, "each NewAdapter call builds its own adapter")

	podID, modelName, _, err := first.ParseMessage(&kvevents.RawMessage{Topic: "pod-1|llama-2-7b"})
	require.NoError(t, err)
	assert.Equal(t, "pod-1", podID)
	assert.Equal(t, "llama-2-7b", modelName)
}

func TestNewAdapter_UnsupportedEngine(t *testing.T) {
	_, err := NewAdapter("own-format")
	require.EqualError(t, err, `unsupported engine type: "own-format" (supported: "sglang", "vllm")`)

	registerForTest(t, "own-format", func() kvevents.EngineAdapter { return &fakeAdapter{} })
	_, err = NewAdapter("other-format")
	require.EqualError(t, err, `unsupported engine type: "other-format" (supported: "own-format", "sglang", "vllm")`)
}

func TestRegister_ReplacesEarlierRegistration(t *testing.T) {
	registerForTest(t, EngineTypeVLLM, func() kvevents.EngineAdapter { return &fakeAdapter{} })

	got, err := NewAdapter("")
	require.NoError(t, err)
	assert.IsType(t, &fakeAdapter{}, got)
}
