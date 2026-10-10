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
	"encoding/base64"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

func writeTestKey(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "hmac-key")
	require.NoError(t, os.WriteFile(path, []byte(base64.RawURLEncoding.EncodeToString(make([]byte, 32))), 0o600))
	return path
}

func TestConfigResolve(t *testing.T) {
	t.Parallel()
	cfg, err := (Config{DeploymentID: "prod-a", HMACKeyFile: writeTestKey(t)}).resolve()
	require.NoError(t, err)
	assert.Equal(t, "prod-a", cfg.deploymentID)
	assert.Len(t, cfg.hmacKey, sessionManagerKeyBytes)
}

func TestConfigRejectsUnsafeInputs(t *testing.T) {
	t.Parallel()
	key := writeTestKey(t)
	valid := Config{DeploymentID: "prod", HMACKeyFile: key}
	tests := []struct {
		name   string
		mutate func(*Config)
	}{
		{"missing deployment", func(c *Config) { c.DeploymentID = "" }},
		{"deployment surrounding whitespace", func(c *Config) { c.DeploymentID = " prod " }},
		{"missing key", func(c *Config) { c.HMACKeyFile = "" }},
		{"deployment above maximum", func(c *Config) {
			c.DeploymentID = strings.Repeat("d", maxConfigurationLength+1)
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			cfg := valid
			test.mutate(&cfg)
			_, err := cfg.resolve()
			require.Error(t, err)
		})
	}
}

func TestConfigAcceptsMaximumDeploymentLength(t *testing.T) {
	t.Parallel()
	cfg, err := (Config{
		DeploymentID: strings.Repeat("d", maxConfigurationLength),
		HMACKeyFile:  writeTestKey(t),
	}).resolve()
	require.NoError(t, err)
	assert.Len(t, cfg.deploymentID, maxConfigurationLength)
}

func TestConfigRejectsWrongKeyLength(t *testing.T) {
	t.Parallel()
	path := filepath.Join(t.TempDir(), "hmac-key")
	require.NoError(t, os.WriteFile(path, []byte(base64.RawURLEncoding.EncodeToString(make([]byte, 31))), 0o600))
	_, err := (Config{DeploymentID: "prod", HMACKeyFile: path}).resolve()
	require.Error(t, err)
}

func TestFactoryRejectsUnknownFields(t *testing.T) {
	t.Parallel()
	raw := []byte(`{"deploymentID":"prod","hmacKeyFile":"unused","correlation":true}`)
	_, err := Factory("sessions", fwkplugin.StrictDecoder(json.RawMessage(raw)), nil)
	require.Error(t, err)
	assert.ErrorContains(t, err, "unknown field")
}
