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

package sessionmanager

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/requestheader/sessionidentity"
)

func TestConfigResolveDefaultsIdentityProducer(t *testing.T) {
	t.Parallel()
	cfg, err := (Config{}).resolve()
	require.NoError(t, err)
	assert.Equal(t, sessionidentity.PluginType, cfg.sessionIdentityProducer)
}

func TestConfigResolveNamedIdentityProducer(t *testing.T) {
	t.Parallel()
	cfg, err := (Config{SessionIdentityProducer: "tenant-identity"}).resolve()
	require.NoError(t, err)
	assert.Equal(t, "tenant-identity", cfg.sessionIdentityProducer)
}

func TestConfigRejectsUnsafeProducerNames(t *testing.T) {
	t.Parallel()
	for _, value := range []string{
		" identity ",
		strings.Repeat("i", maxConfigurationLength+1),
	} {
		t.Run(value[:min(len(value), 20)], func(t *testing.T) {
			t.Parallel()
			_, err := (Config{SessionIdentityProducer: value}).resolve()
			require.Error(t, err)
		})
	}
}

func TestFactoryRejectsUnknownFields(t *testing.T) {
	t.Parallel()
	raw := []byte(`{"unknown":true}`)
	_, err := Factory("sessions", fwkplugin.StrictDecoder(json.RawMessage(raw)), nil)
	require.Error(t, err)
	assert.ErrorContains(t, err, "unknown field")
}
