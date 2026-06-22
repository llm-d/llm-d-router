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

// Package burstprefix provides a request-level data producer that co-locates
// bursts of prompt-sharing requests. Requests arriving within a configurable
// window are assigned jointly: samples that share a prompt are steered to the
// same replica(s) so the shared prefix is prefilled once instead of scattered
// across replicas on a cold cache. It emits PrefixCacheMatchInfo and reuses the
// prefix-cache-scorer (point its prefixMatchInfoProducerName at this producer).
package burstprefix

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/prefixhash"
	tokenproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
)

// produceTimeoutMargin is added to the window so the director's per-producer
// timeout does not cancel a request still waiting for its batch to seal.
const produceTimeoutMargin = 300 * time.Millisecond

var (
	_ requestcontrol.DataProducer         = &dataProducer{}
	_ requestcontrol.TimeoutAwareProducer = &dataProducer{}
)

// dataProducer batches requests within a window and assigns each to a replica
// so prompt-sharing samples co-locate.
type dataProducer struct {
	typedName plugin.TypedName
	config    config
	window    time.Duration
	maxBlocks int
	dk        plugin.DataKey

	mu    sync.Mutex
	batch *batch
}

// TypedName returns the type and name of the plugin.
func (p *dataProducer) TypedName() plugin.TypedName {
	return p.typedName
}

// Produces returns the data produced by the plugin.
func (p *dataProducer) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{p.dk: attrprefix.PrefixCacheMatchInfo{}}
}

// Consumes declares the TokenizedPrompt dependency so the token-producer runs
// before this producer and one is auto-created when none is configured.
func (p *dataProducer) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{
		Required: map[plugin.DataKey]any{tokenproducer.TokenizedPromptDataKey: fwksched.TokenizedPrompt{}},
	}
}

// ProduceTimeout extends the producer timeout to cover the batch window.
func (p *dataProducer) ProduceTimeout() time.Duration {
	return p.window + produceTimeoutMargin
}

// newDataProducer initializes a new burst prefix cache producer.
func newDataProducer(_ context.Context, name string, cfg config) (*dataProducer, error) {
	if cfg.WindowDurationMs <= 0 {
		return nil, fmt.Errorf("invalid configuration: windowDurationMs must be > 0 (current value: %d)", cfg.WindowDurationMs)
	}
	if cfg.MaxPerReplica != unlimitedPerReplica && cfg.MaxPerReplica < 1 {
		return nil, fmt.Errorf("invalid configuration: maxPerReplica must be -1 (unlimited) or >= 1 (current value: %d)", cfg.MaxPerReplica)
	}
	if cfg.BlockSizeTokens <= 0 {
		return nil, fmt.Errorf("invalid configuration: blockSizeTokens must be > 0 (current value: %d)", cfg.BlockSizeTokens)
	}
	if cfg.MaxPrefixTokensToMatch < 0 {
		return nil, fmt.Errorf("invalid configuration: maxPrefixTokensToMatch must be >= 0 (current value: %d)", cfg.MaxPrefixTokensToMatch)
	}

	maxBlocks := defaultMaxPrefixBlocks
	if cfg.MaxPrefixTokensToMatch > 0 {
		maxBlocks = cfg.MaxPrefixTokensToMatch / cfg.BlockSizeTokens
	}

	return &dataProducer{
		typedName: plugin.TypedName{Type: PluginType, Name: name},
		config:    cfg,
		window:    time.Duration(cfg.WindowDurationMs) * time.Millisecond,
		maxBlocks: maxBlocks,
		dk:        attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(name),
	}, nil
}

// Produce collects the request into the current batch window, waits for the
// window to seal, then attaches PrefixCacheMatchInfo reflecting the joint
// assignment: a full match on the assigned replica and zero elsewhere.
func (p *dataProducer) Produce(ctx context.Context, request *fwksched.InferenceRequest, pods []fwksched.Endpoint) error {
	hashes := prefixhash.GetBlockHashes(ctx, request, p.config.BlockSizeTokens, p.maxBlocks)
	e := &entry{hashes: hashes, pods: pods}

	p.mu.Lock()
	if p.batch == nil {
		p.batch = &batch{sealed: make(chan struct{})}
		b := p.batch
		time.AfterFunc(p.window, func() { p.seal(b) })
	}
	b := p.batch
	b.entries = append(b.entries, e)
	p.mu.Unlock()

	select {
	case <-b.sealed:
	case <-ctx.Done():
		// Timed out or cancelled before the batch sealed; leave no affinity.
		return ctx.Err()
	}

	total := totalBlocks(hashes)
	for _, pod := range pods {
		matchLen := 0
		if e.assigned != nil && pod.GetMetadata().NamespacedName == e.assigned.GetMetadata().NamespacedName {
			matchLen = total
		}
		pod.Put(p.dk.String(), attrprefix.NewPrefixCacheMatchInfo(matchLen, total, p.config.BlockSizeTokens))
	}
	return nil
}

// seal finalizes a batch: it detaches the batch so later requests start a fresh
// one, computes the joint assignment, and releases all waiting requests.
func (p *dataProducer) seal(b *batch) {
	p.mu.Lock()
	if b.closed {
		p.mu.Unlock()
		return
	}
	b.closed = true
	if p.batch == b {
		p.batch = nil
	}
	entries := b.entries
	p.mu.Unlock()

	assign(entries, p.config.MaxPerReplica)
	close(b.sealed)
}

// Factory is the factory function for the burst prefix cache producer plugin.
func Factory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := defaultConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&parameters); err != nil {
			return nil, fmt.Errorf("failed to unmarshal burst prefix cache parameters: %w", err)
		}
	}
	if handle == nil {
		return nil, errors.New("plugin handle is required")
	}
	log.FromContext(handle.Context()).V(logutil.DEFAULT).Info("Burst prefix DataProducer initialized", "config", parameters)
	return newDataProducer(handle.Context(), name, parameters)
}
