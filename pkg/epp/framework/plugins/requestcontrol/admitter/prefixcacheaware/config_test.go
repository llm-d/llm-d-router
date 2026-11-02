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

package prefixcacheaware

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

func TestConfigParser(t *testing.T) {
	tests := []struct {
		name    string
		json    string
		wantErr string
		check   func(t *testing.T, c *Config)
	}{
		{
			name: "defaults",
			json: `{"saturationDetectorRef":"det"}`,
			check: func(t *testing.T, c *Config) {
				assert.Equal(t, 0.85, c.ShedThreshold)
				assert.Equal(t, 0.6, c.MinHitRatio)
				assert.Equal(t, int64(0), c.MinUncachedTokens)
				assert.Equal(t, 429, c.ShedStatusCode)
			},
		},
		{name: "missing ref", json: `{}`, wantErr: "saturationDetectorRef is required"},
		{name: "threshold zero", json: `{"saturationDetectorRef":"det","shedThreshold":0}`, wantErr: "shedThreshold"},
		{name: "threshold above one", json: `{"saturationDetectorRef":"det","shedThreshold":1.5}`, wantErr: "shedThreshold"},
		{name: "hit ratio above one", json: `{"saturationDetectorRef":"det","minHitRatio":1.5}`, wantErr: "minHitRatio"},
		{name: "negative min uncached", json: `{"saturationDetectorRef":"det","minUncachedTokens":-1}`, wantErr: "minUncachedTokens"},
		{name: "bad status code", json: `{"saturationDetectorRef":"det","shedStatusCode":500}`, wantErr: "shedStatusCode"},
		{name: "unknown field", json: `{"saturationDetectorRef":"det","nope":1}`, wantErr: "unknown field"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			raw, err := ConfigParser(fwkplugin.StrictDecoder([]byte(tc.json)), nil)
			if tc.wantErr != "" {
				require.Error(t, err)
				assert.Regexp(t, tc.wantErr, err.Error())
				return
			}
			require.NoError(t, err)
			tc.check(t, raw.(*Config))
		})
	}
}

func TestConfigParserNilParameters(t *testing.T) {
	_, err := ConfigParser(nil, nil)
	require.ErrorContains(t, err, "saturationDetectorRef is required")
}
