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

package burstprefix

import (
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixhash"
)

const (
	// PluginType is the unique identifier for the burst prefix cache producer plugin.
	PluginType = "burst-prefix-cache-producer"

	// unlimitedPerReplica is the MaxPerReplica sentinel that disables the
	// per-replica cap, sending every sample of a group to a single replica.
	unlimitedPerReplica = -1

	defaultWindowDurationMs = 100
	defaultMaxPerReplica    = unlimitedPerReplica
	defaultBlockSizeTokens  = 64

	// defaultMaxPrefixBlocks caps how many leading blocks form a group key.
	// Two prompts identical up to this many blocks are treated as one group.
	defaultMaxPrefixBlocks = 2048
)

// config defines the configuration for the burst prefix cache producer.
type config struct {
	// WindowDurationMs is the batch window T in milliseconds. Requests arriving
	// within one window are assigned jointly so samples sharing a prompt
	// co-locate on the same replica(s).
	WindowDurationMs int `json:"windowDurationMs"`
	// MaxPerReplica caps how many samples of one group are assigned to a single
	// replica (k). -1 disables the cap (all samples of a group to one replica).
	MaxPerReplica int `json:"maxPerReplica"`
	// BlockSizeTokens is the token block size used to compute prefix hashes.
	BlockSizeTokens int `json:"blockSizeTokens"`
	// MaxPrefixTokensToMatch caps prefix matching in tokens. When > 0 it sets
	// maxBlocks = MaxPrefixTokensToMatch / BlockSizeTokens; otherwise
	// defaultMaxPrefixBlocks applies.
	MaxPrefixTokensToMatch int `json:"maxPrefixTokensToMatch"`
}

// defaultConfig provides sensible defaults for the burst prefix cache producer.
var defaultConfig = config{
	WindowDurationMs:       defaultWindowDurationMs,
	MaxPerReplica:          defaultMaxPerReplica,
	BlockSizeTokens:        defaultBlockSizeTokens,
	MaxPrefixTokensToMatch: 0,
}

// entry is one request collected into a batch window.
type entry struct {
	hashes [][]prefixhash.BlockHash
	pods   []fwksched.Endpoint
	// assigned is the replica this request is steered to, filled when the batch
	// is sealed. nil means no affinity (singleton group or empty prompt): the
	// request is scored 0 on every endpoint so other scorers decide.
	assigned fwksched.Endpoint
}

// batch accumulates requests arriving within one window and releases them
// together once sealed.
type batch struct {
	entries []*entry
	sealed  chan struct{}
	closed  bool
}
