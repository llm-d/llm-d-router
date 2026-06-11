// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kvevents

import (
	"context"
	"fmt"
	"hash/fnv"
	"strings"
	"sync"

	"k8s.io/client-go/util/workqueue"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/utils/logging"
)

const (
	defaultEventSourceDeviceTier = "gpu"
	defaultPodSelector           = "llm-d.ai/inference-serving=true"
)

// Config holds the configuration for the event processing pool.
type Config struct {
	// ZMQEndpoint is the ZMQ address to connect to (e.g., "tcp://indexer:5557").
	ZMQEndpoint string `json:"zmqEndpoint,omitempty"`
	// TopicFilter is the ZMQ subscription filter (e.g., "kv@").
	TopicFilter string `json:"topicFilter"`
	// Concurrency is the number of parallel workers to run.
	Concurrency int `json:"concurrency"`
	// EngineType selects the inference engine adapter ("vllm" or "sglang").
	// Default: "vllm".
	EngineType string `json:"engineType,omitempty"`
	// DiscoverPods enables the Kubernetes pod reconciler for automatic
	// per-pod subscriber management. When enabled, the reconciler watches
	// Kubernetes pods and creates/removes ZMQ subscribers dynamically.
	DiscoverPods bool `json:"discoverPods"`
	// PodDiscoveryConfig holds the configuration for pod discovery.
	// Only used when DiscoverPods is true.
	PodDiscoveryConfig *PodDiscoveryConfig `json:"podDiscoveryConfig,omitempty"`
}

// PodDiscoveryConfig holds configuration for the Kubernetes pod reconciler.
type PodDiscoveryConfig struct {
	// PodLabelSelector is a label selector string for filtering which pods to watch.
	// Example: "app=vllm" or "app=vllm,tier=gpu"
	PodLabelSelector string `json:"podLabelSelector"`
	// PodNamespace limits the reconciler to watch pods in a specific namespace.
	// If empty, watches all namespaces (requires appropriate RBAC).
	PodNamespace string `json:"podNamespace,omitempty"`
	// SocketPort is the port number where vLLM pods expose their ZMQ socket.
	// The reconciler will connect to tcp://<PodIP>:<SocketPort>
	// Default: 5557
	SocketPort int `json:"socketPort"`
}

// DefaultPodReconcilerConfig returns a default configuration for the pod reconciler.
func DefaultPodReconcilerConfig() *PodDiscoveryConfig {
	return &PodDiscoveryConfig{
		PodLabelSelector: defaultPodSelector,
		SocketPort:       5557,
	}
}

// DefaultConfig returns a default configuration for the event processing pool.
func DefaultConfig() *Config {
	return &Config{
		TopicFilter:        "kv@",
		Concurrency:        4,
		DiscoverPods:       true,
		PodDiscoveryConfig: DefaultPodReconcilerConfig(),
	}
}

// Pool is a sharded worker pool that processes events from ZMQ subscribers.
// It ensures that events for the same PodIdentifier are processed in order.
// Pool is stateless — all key mappings are delegated to the Index.
type Pool struct {
	queues         []workqueue.TypedRateLimitingInterface[*RawMessage]
	concurrency    int // can replace use with len(queues)
	index          kvblock.Index
	tokenProcessor kvblock.TokenProcessor
	adapter        EngineAdapter
	groupCatalog   *kvblock.GroupCatalog
	wg             sync.WaitGroup
}

// NewPool creates a Pool with a sharded worker setup.
// Subscribers are managed by SubscriberManager which is controlled by the pod
// reconciler.
func NewPool(cfg *Config, index kvblock.Index, tokenProcessor kvblock.TokenProcessor,
	adapter EngineAdapter,
) *Pool {
	if cfg == nil {
		cfg = DefaultConfig()
	}

	p := &Pool{
		queues:         make([]workqueue.TypedRateLimitingInterface[*RawMessage], cfg.Concurrency),
		concurrency:    cfg.Concurrency,
		index:          index,
		tokenProcessor: tokenProcessor,
		adapter:        adapter,
		groupCatalog:   kvblock.NewGroupCatalog(),
	}

	for i := 0; i < p.concurrency; i++ {
		p.queues[i] = workqueue.NewTypedRateLimitingQueue(workqueue.DefaultTypedControllerRateLimiter[*RawMessage]())
	}

	return p
}

// GroupCatalog returns the KV cache group metadata learned from events.
func (p *Pool) GroupCatalog() *kvblock.GroupCatalog {
	return p.groupCatalog
}

// Start begins the worker pool.
// It is non-blocking.
func (p *Pool) Start(ctx context.Context) {
	logger := log.FromContext(ctx)
	logger.Info("Starting sharded event processing pool", "workers", p.concurrency)

	p.wg.Add(p.concurrency)
	for i := 0; i < p.concurrency; i++ {
		// Each worker is given its own dedicated queue shard.
		go p.worker(ctx, i)
	}
}

// Shutdown gracefully stops the pool and its global subscriber if present.
func (p *Pool) Shutdown(ctx context.Context) {
	logger := log.FromContext(ctx)
	logger.Info("Shutting down event processing pool...")

	for _, queue := range p.queues {
		queue.ShutDown()
	}

	p.wg.Wait()
	logger.Info("event processing pool shut down.")
}

// AddTask is called by the subscriber to add a message to the processing queue.
// It hashes the sharding key to select a queue, ensuring messages for the
// same pod always go to the same worker (ordered queue).
func (p *Pool) AddTask(task *RawMessage) {
	key := p.adapter.ShardingKey(task)
	// Use an FNV-1a hash to deterministically select a queue.
	h := fnv.New32a()
	_, err := h.Write([]byte(key))
	if err != nil {
		return
	}

	//nolint:gosec // if concurrency overflows then the world is in trouble anyway
	queueIndex := h.Sum32() % uint32(p.concurrency)
	p.queues[queueIndex].Add(task)
}

// worker is the main processing loop for a single worker goroutine.
// It processes messages from its dedicated queue using the workqueue pattern.
func (p *Pool) worker(ctx context.Context, workerIndex int) {
	defer p.wg.Done()
	queue := p.queues[workerIndex]
	for {
		task, shutdown := queue.Get()
		if shutdown {
			return
		}

		// Use a nested func to ensure Done is always called.
		func(task *RawMessage) {
			defer queue.Done(task)
			p.processRawMessage(ctx, task)
			// Task succeeded, remove it from the queue.
			queue.Forget(task)
		}(task)

		// Check if context was cancelled after processing a task.
		select {
		case <-ctx.Done():
			return
		default:
		}
	}
}

// processRawMessage decodes the raw message payload using the adapter and processes the resulting event batch.
func (p *Pool) processRawMessage(ctx context.Context, msg *RawMessage) {
	logger := log.FromContext(ctx)

	podID, modelName, batch, err := p.adapter.ParseMessage(msg)
	if err != nil {
		logger.Error(err, "Failed to parse message")
		return
	}

	p.processEventBatch(ctx, &batch, podID, modelName)
}

// realignExtraFeatures converts per-engine-block extra features to per-canonical-block
// granularity so that len(result) matches the canonical chunk count expected by
// TokensToKVBlockKeys.
//
// For 1:many (engine BS > canonical BS): each engine block's features are replicated
// to all its constituent canonical sub-blocks.
// For many:1 (engine BS < canonical BS): features from multiple engine blocks are
// merged (union of MMHashes) into each canonical block.
//
// When all entries are nil (text-only prompts), this simply produces a nil-filled
// slice of the correct length.
func realignExtraFeatures(engineFeatures []*kvblock.BlockExtraFeatures, canonicalBlockCount int) []*kvblock.BlockExtraFeatures {
	engineBlockCount := len(engineFeatures)
	if canonicalBlockCount == 0 {
		return nil
	}
	if engineBlockCount == 0 || engineBlockCount == canonicalBlockCount {
		return engineFeatures
	}

	canonical := make([]*kvblock.BlockExtraFeatures, canonicalBlockCount)

	if engineBlockCount < canonicalBlockCount {
		// 1:many -> replicate each engine feature to its canonical sub-blocks
		for i := range canonicalBlockCount {
			engineIdx := i * engineBlockCount / canonicalBlockCount
			canonical[i] = engineFeatures[engineIdx]
		}
	} else {
		// many:1 -> merge constituent engine features into each canonical block
		for i, ef := range engineFeatures {
			canonicalIdx := i * canonicalBlockCount / engineBlockCount
			if ef == nil {
				continue
			}
			if canonical[canonicalIdx] == nil {
				canonical[canonicalIdx] = &kvblock.BlockExtraFeatures{}
			}
			canonical[canonicalIdx].MMHashes = append(
				canonical[canonicalIdx].MMHashes, ef.MMHashes...)
		}
	}

	return canonical
}

// handleDeviceTierUpdate handles offloading/location-only events (e.g., DeviceTier=CPU
// with no tokens). It resolves existing request keys from the engine→request mapping and
// adds the new PodEntry so the EPP tracks which device tiers hold each block.
func (p *Pool) handleDeviceTierUpdate(
	ctx context.Context, tokens []uint32, engineKeys []kvblock.BlockHash,
	podEntries []kvblock.PodEntry, podIdentifier, deviceTier string,
) {
	debugLogger := log.FromContext(ctx).V(logging.DEBUG)

	// Only attempt resolution when tokens are truly absent; partial-block
	// events (tokens < blockSize) should just be skipped.
	if len(tokens) != 0 || len(engineKeys) == 0 {
		return
	}

	seen := make(map[kvblock.BlockHash]struct{})
	var resolvedKeys []kvblock.BlockHash
	for _, ek := range engineKeys {
		rk, err := p.index.GetRequestKey(ctx, ek)
		if err != nil {
			continue
		}
		if _, ok := seen[rk]; !ok {
			seen[rk] = struct{}{}
			resolvedKeys = append(resolvedKeys, rk)
		}
	}

	if len(resolvedKeys) > 0 {
		if err := p.index.Add(ctx, nil, resolvedKeys, podEntries); err != nil {
			debugLogger.Error(err, "Failed to add device-tier update to index",
				"podIdentifier", podIdentifier, "deviceTier", deviceTier)
		}
	} else {
		debugLogger.Info("no indexed engine keys found for device-tier update, skipping",
			"podIdentifier", podIdentifier, "engineKeyCount", len(engineKeys))
	}
}

// processEventBatch processes a batch of events using type switches.
func (p *Pool) processEventBatch(ctx context.Context, batch *EventBatch, podIdentifier, modelName string) {
	debugLogger := log.FromContext(ctx).V(logging.DEBUG)
	debugLogger.V(logging.TRACE).Info("Processing event batch",
		"podID", podIdentifier,
		"modelName", modelName,
		"eventCount", len(batch.Events))

	// Process each event in the batch
	for _, genericEvent := range batch.Events {
		switch ev := genericEvent.(type) {
		case *BlockStoredEvent:
			// Default to gpu.
			deviceTier := defaultEventSourceDeviceTier
			if ev.DeviceTier != "" {
				deviceTier = strings.ToLower(ev.DeviceTier)
			}

			// Use LoRA name as model identifier if available, otherwise fall back to base model name.
			effectiveModelName := modelName
			if ev.LoraName != nil && *ev.LoraName != "" {
				effectiveModelName = *ev.LoraName
			}

			// Create PodEntry for this specific event's device tier.
			podEntries := []kvblock.PodEntry{{PodIdentifier: podIdentifier, DeviceTier: deviceTier}}
			if ev.GroupIdx != nil {
				g := kvblock.GroupID(*ev.GroupIdx)
				p.groupCatalog.Learn(podIdentifier, g, kvblock.GroupMetadata{
					Kind:              string(ev.KVCacheSpecKind),
					BlockSize:         ev.BlockSize,
					SlidingWindowSize: ev.KVCacheSpecSlidingWindowSize,
				})
				podEntries[0].HasGroup = true
				podEntries[0].GroupIdx = g
			}

			engineKeys := make([]kvblock.BlockHash, len(ev.BlockHashes))
			for i, hash := range ev.BlockHashes {
				engineKeys[i] = kvblock.BlockHash(hash)
			}

			parentRequestKey := kvblock.EmptyBlockHash
			if ev.ParentHash != 0 {
				parentEngineKey := kvblock.BlockHash(ev.ParentHash)
				key, err := p.index.GetRequestKey(ctx, parentEngineKey)
				if err != nil {
					debugLogger.Error(err, "Failed to get request key for parent block",
						"parentEngineKey", parentEngineKey, "effectiveModelName", effectiveModelName)
					continue
				}
				parentRequestKey = key
			}

			var extraFeatures []*kvblock.BlockExtraFeatures
			if ev.ExtraKeys != nil {
				var err error
				extraFeatures, err = kvblock.ParseRawExtraKeys(ev.ExtraKeys)
				if err != nil {
					debugLogger.Error(err, "Failed to parse extra keys",
						"podIdentifier", podIdentifier)
					continue
				}
			}

			// Realign extraFeatures from engine-block granularity to canonical-block
			// granularity. ParseRawExtraKeys returns one entry per engine block, but
			// TokensToKVBlockKeys expects one entry per canonical block.
			if extraFeatures != nil {
				canonicalBlockCount := len(ev.Tokens) / p.tokenProcessor.BlockSize()
				if canonicalBlockCount == 0 {
					// Tokens don't fill a complete canonical block; no realignment needed
					// since TokensToKVBlockKeys will produce zero keys anyway.
					extraFeatures = nil
				} else if len(extraFeatures) != canonicalBlockCount {
					extraFeatures = realignExtraFeatures(extraFeatures, canonicalBlockCount)
				}
			}

			traceLogger := log.FromContext(ctx).V(logging.TRACE)
			if traceLogger.Enabled() {
				nonNil := 0
				for _, ef := range extraFeatures {
					if ef != nil {
						nonNil++
					}
				}
				traceLogger.Info("BlockStored extra_features",
					"podIdentifier", podIdentifier,
					"hasExtraKeys", ev.ExtraKeys != nil,
					"parsedBlockCount", len(extraFeatures),
					"nonNilBlocks", nonNil,
					"numTokens", len(ev.Tokens),
					"numEngineKeys", len(ev.BlockHashes))
				for bIdx, ef := range extraFeatures {
					if ef != nil {
						traceLogger.Info("BlockStored block extra",
							"podIdentifier", podIdentifier,
							"blockIdx", bIdx,
							"mmHashes", fmt.Sprintf("%+v", ef.MMHashes))
					}
				}
			}

			// Compute request keys at canonical block size (= BlockSize)
			requestKeys, err := p.tokenProcessor.TokensToKVBlockKeys(
				parentRequestKey, ev.Tokens, effectiveModelName, extraFeatures)
			if err != nil {
				debugLogger.Error(err, "Failed to generate request keys",
					"podIdentifier", podIdentifier, "effectiveModelName", effectiveModelName)
				continue
			}

			if len(requestKeys) == 0 {
				p.handleDeviceTierUpdate(ctx, ev.Tokens, engineKeys, podEntries, podIdentifier, deviceTier)
				continue
			}

			// Index.Add infers the engine->request mapping from the ratio of
			// len(engineKeys) to len(requestKeys) (1:1, many:1, or 1:many).
			if err := p.index.Add(ctx, engineKeys, requestKeys, podEntries); err != nil {
				debugLogger.Error(err, "Failed to add event to index",
					"podIdentifier", podIdentifier, "event", ev)
				continue
			}

		case *BlockRemovedEvent:
			// Default to gpu.
			deviceTier := defaultEventSourceDeviceTier
			if ev.DeviceTier != "" {
				deviceTier = strings.ToLower(ev.DeviceTier)
			}

			// Create PodEntry for this specific event's device tier.
			podEntries := []kvblock.PodEntry{{PodIdentifier: podIdentifier, DeviceTier: deviceTier}}
			if ev.GroupIdx != nil {
				podEntries[0].HasGroup = true
				podEntries[0].GroupIdx = kvblock.GroupID(*ev.GroupIdx)
			}

			// Iterate over the hashes and evict each key.
			// The Index handles engine->request key resolution internally for both
			// 1:1 (legacy) and 1:many (canonical) mappings.
			for _, hash := range ev.BlockHashes {
				engineKey := kvblock.BlockHash(hash)
				if err := p.index.Evict(ctx, engineKey, kvblock.EngineKey, podEntries); err != nil {
					debugLogger.Error(err, "Failed to evict engine key from index",
						"podIdentifier", podIdentifier, "engineKey", engineKey)
					continue
				}
			}

		case *AllBlocksClearedEvent:
			debugLogger.Info("All blocks cleared event received",
				"podIdentifier", podIdentifier,
				"deviceTier", ev.DeviceTier,
				"modelName", modelName)

		default:
			debugLogger.Info("Unknown event", "podIdentifier", podIdentifier, "event", genericEvent)
		}
	}
}
