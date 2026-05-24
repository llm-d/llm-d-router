/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
*/

package preciseprefixcache

import (
	"encoding/json"
	"testing"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/stretchr/testify/require"
)

// Regression test for #1157. The precise-prefix-cache plugin wraps the
// external kvblock.TokenProcessorConfig whose block-size field is named
// "blockSize". Users coming from the approximate-prefix-cache plugin (which
// names the same concept "blockSizeTokens") would set blockSizeTokens at the
// top level of this plugin's config and the value would be silently dropped,
// leaving the scorer with the default 16-token block size.
//
// The fix is to accept blockSizeTokens as a top-level alias that overrides
// TokenProcessorConfig.BlockSize when set, so a single config-field name
// works across both prefix-cache plugins.
func TestPluginConfig_BlockSizeTokensAlias(t *testing.T) {
	cases := []struct {
		name          string
		raw           string
		wantBlockSize int
	}{
		{
			name:          "top-level blockSizeTokens sets BlockSize",
			raw:           `{"blockSizeTokens":64}`,
			wantBlockSize: 64,
		},
		{
			name:          "legacy tokenProcessorConfig.blockSize still works",
			raw:           `{"tokenProcessorConfig":{"blockSize":64}}`,
			wantBlockSize: 64,
		},
		{
			name:          "top-level blockSizeTokens wins over legacy",
			raw:           `{"blockSizeTokens":128,"tokenProcessorConfig":{"blockSize":64}}`,
			wantBlockSize: 128,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var cfg PluginConfig
			require.NoError(t, json.Unmarshal([]byte(tc.raw), &cfg))
			cfg.applyAliases()
			require.NotNil(t, cfg.TokenProcessorConfig)
			require.Equal(t, tc.wantBlockSize, cfg.TokenProcessorConfig.BlockSize)
		})
	}
}

// Confirms that a default-constructed config (no TokenProcessorConfig, no
// BlockSizeTokens) leaves resolution to New(), which falls back to
// kvblock.DefaultTokenProcessorConfig().
func TestPluginConfig_NoAliasesPreservesDefaults(t *testing.T) {
	var cfg PluginConfig
	cfg.applyAliases()
	// No top-level BlockSizeTokens means we don't touch TokenProcessorConfig;
	// New() handles the nil case by installing the kvblock default.
	require.Nil(t, cfg.TokenProcessorConfig)
	require.Equal(t, 16, kvblock.DefaultTokenProcessorConfig().BlockSize)
}
