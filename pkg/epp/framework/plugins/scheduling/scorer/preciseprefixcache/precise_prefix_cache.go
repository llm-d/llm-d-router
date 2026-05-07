package preciseprefixcache

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"slices"
	"sync/atomic"
	"time"

	"github.com/jellydator/ttlcache/v3"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents"
	"github.com/llm-d/llm-d-kv-cache/pkg/kvevents/engineadapter"
	"github.com/llm-d/llm-d-kv-cache/pkg/tokenization/types"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
	"k8s.io/apimachinery/pkg/util/sets"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/requestcontrol/dataproducer/tokenizer"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/telemetry"
)

const (
	// PrecisePrefixCachePluginType is the type-name of the Scorer plugin.
	PrecisePrefixCachePluginType = "precise-prefix-cache-scorer"

	// defaultSpeculativeTTL is the default TTL for speculative entries.
	// This should be just long enough to cover the blind spot between
	// routing decision and KV event arrival, maintaining high confidence
	// in speculations while avoiding stale routing affinity.
	defaultSpeculativeTTL = 2 * time.Second

	// stateKey is the PluginState key used to share data between
	// Produce, Score, and PreRequest.
	stateKey = plugin.StateKey("prefix-cache-state")

	// experimentalPrefillProfile is the profile name for P/D disaggregation mode.
	experimentalPrefillProfile = "prefill"
)

type kvCacheIndexer interface {
	GetPodScores(ctx context.Context, renderReq *types.RenderChatRequest, prompt, modelName string, podIdentifiers []string) (map[string]float64, error)
	ScoreTokens(ctx context.Context, tokens []uint32, modelName string, podIdentifiers []string, extraFeatures []*kvblock.BlockExtraFeatures) (map[string]float64, error)
	ComputeBlockKeys(ctx context.Context, renderReq *types.RenderChatRequest, prompt, modelName string) ([]kvblock.BlockHash, error)
	ComputeBlockKeysFromTokens(ctx context.Context, tokens []uint32, modelName string, extraFeatures []*kvblock.BlockExtraFeatures) ([]kvblock.BlockHash, error)
	KVBlockIndex() kvblock.Index
}

// PluginConfig holds the configuration for the
// Scorer plugin.
type PluginConfig struct {
	// TokenProcessorConfig holds the configuration for the `kvblock.TokenProcessor` which is
	// used to process tokens into KV-block keys.
	TokenProcessorConfig *kvblock.TokenProcessorConfig `json:"tokenProcessorConfig"`
	// IndexerConfig holds the configuration for the `kvcache.Indexer` which is
	// used to score endpoints based on the KV-cache index state.
	IndexerConfig *kvcache.Config `json:"indexerConfig"`
	// KVEventsConfig holds the configuration for the `kvevents.Pool` which is
	// used to subscribe to KV-cache events and update the internal KV-cache
	// index state.
	KVEventsConfig *kvevents.Config `json:"kvEventsConfig"`
	// SpeculativeIndexing enables speculative indexing. When true, the plugin
	// proactively adds predicted cache entries to the index immediately after
	// a routing decision (via Produce and PreRequest), closing the
	// blind spot between routing and KV event arrival.
	// When false, only confirmed KV events populate the index.
	SpeculativeIndexing bool `json:"speculativeIndexing"`
	// SpeculativeTTL is the time-to-live for speculative index entries.
	// After this duration, speculative entries are evicted from the index.
	// If empty, defaultSpeculativeTTL is used. Only used when SpeculativeIndexing is true.
	// Accepts Go duration strings (e.g. "2s", "500ms").
	SpeculativeTTL string `json:"speculativeTTL"`
}

// compile-time type assertions
var (
	_ scheduling.Scorer           = &Scorer{}
	_ requestcontrol.DataProducer = &Scorer{}
	_ requestcontrol.PreRequest   = &Scorer{}
	_ fwkdl.EndpointExtractor     = &Scorer{}
)

// speculativeEntries holds the data needed to evict speculative entries
// from the index when the TTL expires.
type speculativeEntries struct {
	blockKeys  []kvblock.BlockHash
	podEntries []kvblock.PodEntry
}

// precisePluginState holds data shared between Produce, Score,
// and PreRequest via PluginState.
type precisePluginState struct {
	blockKeys [][]kvblock.BlockHash // per-prompt block keys; single prompt = one element
	scores    map[string]float64    // pod addr → score
}

// Clone implements plugin.StateData.
func (s *precisePluginState) Clone() plugin.StateData {
	blockKeys := make([][]kvblock.BlockHash, len(s.blockKeys))
	for i, keys := range s.blockKeys {
		cp := make([]kvblock.BlockHash, len(keys))
		copy(cp, keys)
		blockKeys[i] = cp
	}
	scores := make(map[string]float64, len(s.scores))
	for k, v := range s.scores {
		scores[k] = v
	}
	return &precisePluginState{
		blockKeys: blockKeys,
		scores:    scores,
	}
}

// PluginFactory defines the factory function for creating
// a new instance of the PrefixCacheTrackingPlugin.
func PluginFactory(name string, rawParameters json.RawMessage,
	handle plugin.Handle,
) (plugin.Plugin, error) {
	indexerConfig, err := kvcache.NewDefaultConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to initialize indexer config: %w", err)
	}

	parameters := PluginConfig{
		IndexerConfig:  indexerConfig,
		KVEventsConfig: kvevents.DefaultConfig(),
	}

	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse %s plugin config: %w", PrecisePrefixCachePluginType, err)
		}
	}

	if parameters.IndexerConfig == nil {
		return nil, errors.New("indexerConfig is required")
	}
	//nolint:staticcheck // SA1019: validate the legacy field when callers explicitly opt into it.
	if parameters.IndexerConfig.TokenizersPoolConfig != nil && parameters.IndexerConfig.TokenizersPoolConfig.ModelName == "" {
		return nil, errors.New("modelName is required when indexerConfig.tokenizersPoolConfig is set")
	}

	scorer, err := New(handle.Context(), parameters)
	if err != nil {
		return nil, fmt.Errorf("failed to create %s plugin: %w", PrecisePrefixCachePluginType, err)
	}

	return scorer.WithName(name), nil
}

// New initializes a new prefix Plugin and returns its pointer.
// It sets up the `kvcache.Indexer` and `kvevents.Pool`
// based on the provided configuration. The `kvevents.Pool` is started
// in a goroutine to listen for KV-cache events and update the internal
// KV-cache index state. The `kvcache.Indexer` is also started in a goroutine
// to score endpoints based on the KV-cache index state.
//
// If the configuration is invalid or if the indexer fails to initialize,
// an error is returned.
func New(ctx context.Context, config PluginConfig) (*Scorer, error) {
	if config.TokenProcessorConfig == nil {
		config.TokenProcessorConfig = kvblock.DefaultTokenProcessorConfig()
	}

	tokenProcessor, err := kvblock.NewChunkedTokenDatabase(config.TokenProcessorConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create token processor: %w", err)
	}

	// initialize the indexer
	kvCacheIndexer, err := kvcache.NewKVCacheIndexer(ctx, config.IndexerConfig, tokenProcessor)
	if err != nil {
		return nil, fmt.Errorf("failed to create `kvcache.Indexer`: %w", err)
	}

	go kvCacheIndexer.Run(ctx)

	// initialize the KV block scorer with the same config the indexer uses
	scorerConfig := kvcache.DefaultKVBlockScorerConfig()
	if config.IndexerConfig != nil && config.IndexerConfig.BackendConfigs != nil {
		scorerConfig.BackendConfigs = config.IndexerConfig.BackendConfigs
	}
	kvBlockScorer, err := kvcache.NewKVBlockScorer(scorerConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create KVBlockScorer: %w", err)
	}

	// initialize the KV-events pool
	pool := kvevents.NewPool(config.KVEventsConfig, kvCacheIndexer.KVBlockIndex(), tokenProcessor, engineadapter.NewVLLMAdapter())
	pool.Start(ctx)

	subscribersManager := kvevents.NewSubscriberManager(pool)

	if config.KVEventsConfig.ZMQEndpoint != "" {
		// setup local subscriber to support global socket mode
		if err := subscribersManager.EnsureSubscriber(ctx, "local-subscriber",
			config.KVEventsConfig.ZMQEndpoint, config.KVEventsConfig.TopicFilter, false); err != nil {
			return nil, fmt.Errorf("failed to create local subscriber for global socket mode: %w", err)
		}
	}

	// Initialize speculative indexing components only when enabled
	var speculativeCache *ttlcache.Cache[string, *speculativeEntries]
	var speculativeTTL time.Duration
	if config.SpeculativeIndexing {
		if config.SpeculativeTTL != "" {
			var err error
			speculativeTTL, err = time.ParseDuration(config.SpeculativeTTL)
			if err != nil {
				return nil, fmt.Errorf("invalid speculativeTTL %q: %w", config.SpeculativeTTL, err)
			}
		}
		if speculativeTTL <= 0 {
			speculativeTTL = defaultSpeculativeTTL
		}

		speculativeCache = ttlcache.New[string, *speculativeEntries](
			ttlcache.WithTTL[string, *speculativeEntries](speculativeTTL),
		)
		speculativeCache.OnEviction(func(_ context.Context, reason ttlcache.EvictionReason,
			item *ttlcache.Item[string, *speculativeEntries],
		) {
			if reason != ttlcache.EvictionReasonExpired {
				return
			}
			entries := item.Value()
			for _, reqKey := range entries.blockKeys {
				// Evict speculative entries from the index.
				// Speculative entries were added without engineKey mapping (nil engineKeys),
				// so we use RequestKey type to evict by requestKey directly.
				//nolint:errcheck // best-effort cleanup on TTL expiry
				kvCacheIndexer.KVBlockIndex().Evict(context.Background(), reqKey, kvblock.RequestKey, entries.podEntries)
			}
		})
		go cleanCachePeriodically(ctx, speculativeCache, speculativeTTL)
	}

	return &Scorer{
		typedName:          plugin.TypedName{Type: PrecisePrefixCachePluginType},
		kvCacheIndexer:     kvCacheIndexer,
		kvBlockScorer:      kvBlockScorer,
		subscribersManager: subscribersManager,
		kvEventsConfig:     config.KVEventsConfig,
		pluginState:        plugin.NewPluginState(ctx),
		speculativeCache:   speculativeCache,
		speculativeTTL:     speculativeTTL,
		blockSizeTokens:    config.TokenProcessorConfig.BlockSize,
		speculativeEnabled: config.SpeculativeIndexing,
		subscriberCtx:      ctx,
	}, nil
}

// Scorer implements the framework.Scorer interface.
// The scorer implements precise prefix-cache KV-block locality scoring.
// It uses the `kvcache.Indexer` to score endpoints based on the KV-cache index
// state, and the `kvevents.Pool` to subscribe to KV-cache events
// to keep the internal KV-cache index state up-to-date.
//
// With speculative indexing, the scorer also implements DataProducerPlugin and
// PreRequest to proactively populate the index with expected cache entries
// immediately after a routing decision, closing the blind spot between the
// routing decision and the arrival of actual KV events from the engine.
//
// The scorer also implements EndpointExtractor to react to endpoint lifecycle
// events from the data layer's endpoint-notification-source: an add/update
// installs a per-pod ZMQ subscriber, a delete removes it.
type Scorer struct {
	typedName      plugin.TypedName
	kvCacheIndexer kvCacheIndexer

	subscribersManager *kvevents.SubscriberManager
	kvEventsConfig     *kvevents.Config

	// pluginState stores per-request data (block keys, scores) shared
	// between Produce, Score, and PreRequest extension points.
	pluginState *plugin.PluginState

	// speculativeCache tracks speculative entries added to the index so that
	// they can be evicted when their TTL expires.
	speculativeCache *ttlcache.Cache[string, *speculativeEntries]
	speculativeTTL   time.Duration

	// kvBlockScorer scores pods based on block hits with device-backend weights.
	kvBlockScorer kvcache.KVBlockScorer

	// blockSizeTokens is the number of tokens per KV-block, used for
	// constructing PrefixCacheMatchInfo in Produce.
	blockSizeTokens int

	// speculativeEnabled controls whether speculative indexing is active.
	speculativeEnabled bool

	// extractorActive is set the first time ExtractEndpoint is invoked,
	// signalling that the data layer's endpoint-notification-source is wired
	// for this scorer. Once set, the legacy in-Score subscriber discovery
	// path becomes a no-op so the data layer is the sole authority over
	// per-pod subscriber lifecycle.
	extractorActive atomic.Bool

	// subscriberCtx is the long-lived context used to start ZMQ subscribers.
	// SubscriberManager binds each subscriber's goroutine lifetime to the
	// context passed in to EnsureSubscriber, so any caller-scoped context
	// (e.g. the request ctx in the legacy in-Score path) would tear the
	// subscriber down as soon as the caller returned. Using the plugin's
	// construction-time ctx keeps subscribers alive for the EPP's lifetime,
	// matching the original behavior of `context.Background()` in the
	// pre-refactor code.
	subscriberCtx context.Context
}

// TypedName returns the typed name of the plugin.
func (s *Scorer) TypedName() plugin.TypedName {
	return s.typedName
}

// WithName sets the name of the plugin.
func (s *Scorer) WithName(name string) *Scorer {
	s.typedName.Name = name
	return s
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *Scorer) Category() scheduling.ScorerCategory {
	return scheduling.Affinity
}

// --- DataProducerPlugin implementation ---

// Produces declares the data keys this plugin writes to endpoints.
func (s *Scorer) Produces() map[string]any {
	return map[string]any{
		attrprefix.PrefixCacheMatchInfoKey: attrprefix.PrefixCacheMatchInfo{},
	}
}

// Produce computes block keys, looks up the index, and stores
// per-endpoint prefix match information. The computed block keys and scores
// are saved to PluginState for reuse by Score() and PreRequest().
// This is a no-op when speculative indexing is disabled.
func (s *Scorer) Produce(ctx context.Context,
	request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) error {
	if !s.speculativeEnabled {
		return nil
	}

	logger := log.FromContext(ctx).WithName(s.typedName.String())

	if request == nil || request.Body == nil {
		return nil
	}

	// Build pod set from endpoints
	podSet := extractPodSet(endpoints)

	if request.Body.Completions != nil && len(request.Body.Completions.Prompt.Strings) > 0 {
		var perPromptScores []map[string]float64
		var perPromptKeys [][]kvblock.BlockHash
		totalBlocks := 0

		// Keep string-array completions as separate prefix chains. Flattening
		// here would create cross-prompt block adjacency that never exists in
		// the model server cache and would poison speculative indexing.
		for _, promptStr := range request.Body.Completions.Prompt.Strings {
			keys, err := s.kvCacheIndexer.ComputeBlockKeys(ctx, nil, promptStr, request.TargetModel)
			if err != nil {
				return fmt.Errorf("failed to compute block keys: %w", err)
			}
			perPromptKeys = append(perPromptKeys, keys)
			totalBlocks += len(keys)

			if len(keys) == 0 {
				continue
			}

			keyToPods, err := s.kvCacheIndexer.KVBlockIndex().Lookup(ctx, keys, podSet)
			if err != nil {
				return fmt.Errorf("failed to lookup block keys: %w", err)
			}
			scores, err := s.kvBlockScorer.Score(ctx, keys, keyToPods)
			if err != nil {
				return fmt.Errorf("failed to score block keys: %w", err)
			}
			perPromptScores = append(perPromptScores, scores)
		}

		if totalBlocks == 0 {
			return nil
		}

		aggregatedScores := aggregatePromptScores(perPromptScores)
		blockSize := s.getBlockSizeTokens()
		for _, ep := range endpoints {
			md := ep.GetMetadata()
			if md == nil {
				continue
			}
			addr := fmt.Sprintf("%s:%s", md.Address, md.Port)
			matchLen := int(aggregatedScores[addr])
			ep.Put(attrprefix.PrefixCacheMatchInfoKey,
				attrprefix.NewPrefixCacheMatchInfo(matchLen, totalBlocks, blockSize))
		}

		s.pluginState.Write(request.RequestID, stateKey, &precisePluginState{
			blockKeys: perPromptKeys,
			scores:    aggregatedScores,
		})
		return nil
	}

	// 1. Compute block keys from the request
	blockKeys, err := s.computeBlockKeys(ctx, request)
	if err != nil {
		return fmt.Errorf("failed to compute block keys: %w", err)
	}
	if len(blockKeys) == 0 {
		return nil
	}

	// 2. Lookup index for matching pods
	keyToPods, err := s.kvCacheIndexer.KVBlockIndex().Lookup(ctx, blockKeys, podSet)
	if err != nil {
		return fmt.Errorf("failed to lookup block keys: %w", err)
	}

	// 3. Compute per-pod scores using KVBlockScorer (supports device-backend weights)
	scores, err := s.kvBlockScorer.Score(ctx, blockKeys, keyToPods)
	if err != nil {
		return fmt.Errorf("failed to score block keys: %w", err)
	}

	// 4. Store PrefixCacheMatchInfo on each endpoint
	blockSize := s.getBlockSizeTokens()
	for _, ep := range endpoints {
		md := ep.GetMetadata()
		if md == nil {
			continue
		}
		addr := fmt.Sprintf("%s:%s", md.Address, md.Port)
		matchLen := int(scores[addr])
		ep.Put(attrprefix.PrefixCacheMatchInfoKey, attrprefix.NewPrefixCacheMatchInfo(matchLen, len(blockKeys), blockSize))
	}

	// 6. Save to PluginState for Score() and PreRequest()
	s.pluginState.Write(request.RequestID, stateKey, &precisePluginState{
		blockKeys: [][]kvblock.BlockHash{blockKeys},
		scores:    scores,
	})

	logger.V(logging.TRACE).Info("Produce completed",
		"blockKeys", len(blockKeys), "scores", scores)

	return nil
}

// --- Scorer implementation ---

// Score scores the provided endpoint based on the KVCache index state.
// The returned scores are normalized to a range of 0-1.
// If Produce was called beforehand, Score reuses the pre-computed
// results from PluginState. Otherwise, it falls back to computing scores
// directly via getScores (backward compatible).
func (s *Scorer) Score(ctx context.Context, cycleState *scheduling.CycleState, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	// Start tracing span for scoring operation
	tracer := telemetry.Tracer()
	ctx, span := tracer.Start(ctx, "llm_d.epp.scorer.prefix_cache",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer span.End()

	logger := log.FromContext(ctx).WithName(s.typedName.String())
	debugLogger := logger.V(logging.DEBUG)

	// Set initial attributes
	span.SetAttributes(
		attribute.Int("llm_d.scorer.candidate_endpoints", len(endpoints)),
	)

	// Backwards-compat: opportunistically subscribe to per-pod KV events for
	// each endpoint we see in scoring. Preferred path is the data-layer
	// EndpointExtractor (see ExtractEndpoint), which also handles teardown
	// when pods disappear. This in-Score path keeps existing configs that
	// don't wire the endpoint-notification-source working without changes;
	// EnsureSubscriber is idempotent so the two paths are safe to run
	// together.
	s.ensureSubscribersForEndpoints(ctx, endpoints)

	// Early return if request is nil
	if request == nil {
		debugLogger.Info("Request is nil, skipping scoring")
		span.SetAttributes(attribute.String("llm_d.scorer.result", "skipped_nil_request"))
		return nil
	}

	// Set optional request attributes
	if request.TargetModel != "" {
		span.SetAttributes(attribute.String("gen_ai.request.model", request.TargetModel))
	}
	if request.RequestID != "" {
		span.SetAttributes(attribute.String("gen_ai.request.id", request.RequestID))
	}

	// Try to reuse pre-computed scores from Produce
	var scores map[string]float64
	if pluginStateData, err := plugin.ReadPluginStateKey[*precisePluginState](
		s.pluginState, request.RequestID, stateKey); err == nil {
		scores = pluginStateData.scores
		debugLogger.Info("Reusing pre-computed scores from Produce")
	} else {
		// Fallback: compute scores directly (backward compatible path).
		var scoreErr error
		scores, scoreErr = s.getScores(ctx, cycleState, request)
		if scoreErr != nil {
			logger.Error(scoreErr, "Failed to get endpoint scores")
			span.SetStatus(codes.Error, scoreErr.Error())
			return nil
		}
	}
	debugLogger.Info("Got endpoint scores", "scores", scores)

	// Track scoring statistics
	span.SetAttributes(
		attribute.Int("llm_d.scorer.scores_computed", len(scores)),
	)

	endpointToKey := func(endpoint scheduling.Endpoint) (string, bool) {
		metadata := endpoint.GetMetadata()
		if metadata == nil {
			return "", false
		}

		return fmt.Sprintf("%s:%s", metadata.Address, metadata.Port), true
	}

	// Write per-endpoint prefix-cache match info as endpoint attributes so downstream
	// scorers (e.g. nohitlru) can determine whether any cache hits were found.
	for _, endpoint := range endpoints {
		key, ok := endpointToKey(endpoint)
		matchBlocks := 0
		if ok {
			if rawScore, exists := scores[key]; exists && rawScore > 0 {
				matchBlocks = 1
			}
		}
		endpoint.Put(attrprefix.PrefixCacheMatchInfoKey, attrprefix.NewPrefixCacheMatchInfo(matchBlocks, 1, 1))
	}

	normalizedScores := indexedScoresToNormalizedScoredPods(endpoints, endpointToKey, scores)

	// Calculate score distribution for observability
	if len(normalizedScores) > 0 {
		maxScore := 0.0
		totalScore := 0.0
		for _, score := range normalizedScores {
			if score > maxScore {
				maxScore = score
			}
			totalScore += score
		}
		avgScore := totalScore / float64(len(normalizedScores))

		span.SetAttributes(
			attribute.Float64("llm_d.scorer.score.max", maxScore),
			attribute.Float64("llm_d.scorer.score.avg", avgScore),
			attribute.Int("llm_d.scorer.endpoints_scored", len(normalizedScores)),
		)
	}

	return normalizedScores
}

// --- PreRequest implementation ---

// PreRequest records speculative entries in the index for the selected endpoint
// immediately after the scheduling decision. This closes the blind spot between
// the routing decision and the arrival of actual KV events from the engine.
// The speculative entries are associated with a TTL and will be automatically
// evicted when the TTL expires.
// This is a no-op when speculative indexing is disabled.
func (s *Scorer) PreRequest(ctx context.Context,
	request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	if !s.speculativeEnabled {
		return
	}

	logger := log.FromContext(ctx).WithName(s.typedName.String())

	// 1. Read block keys from PluginState
	state, err := plugin.ReadPluginStateKey[*precisePluginState](
		s.pluginState, request.RequestID, stateKey)
	if err != nil {
		logger.V(logging.TRACE).Info("No plugin state found for PreRequest, skipping speculative indexing",
			"requestID", request.RequestID)
		return
	}
	s.pluginState.Delete(request.RequestID)

	if len(state.blockKeys) == 0 {
		return
	}

	// 2. Get target endpoint from scheduling result
	primaryResult := schedulingResult.ProfileResults[schedulingResult.PrimaryProfileName]
	if primaryResult == nil || len(primaryResult.TargetEndpoints) == 0 {
		return
	}
	targetEndpoint := primaryResult.TargetEndpoints[0]

	// 3. Build speculative pod entry and add to index
	targetMeta := targetEndpoint.GetMetadata()
	speculativePod := kvblock.PodEntry{
		PodIdentifier: fmt.Sprintf("%s:%s", targetMeta.Address, targetMeta.Port),
		Speculative:   true,
	}

	allPodEntries := []kvblock.PodEntry{speculativePod}

	index := s.kvCacheIndexer.KVBlockIndex()
	// Pass nil engineKeys: speculative entries only need requestKey -> PodEntry mapping.
	// Engine keys will be linked later when confirmed KV events arrive.
	// Each prompt's block keys are added as an independent prefix chain.
	for _, keys := range state.blockKeys {
		if len(keys) == 0 {
			continue
		}
		if err := index.Add(ctx, nil, keys, []kvblock.PodEntry{speculativePod}); err != nil {
			logger.Error(err, "Failed to add speculative entries to index",
				"pod", speculativePod.PodIdentifier)
		}
	}

	// 4. Handle P/D disaggregation: also add speculative entry for prefill endpoint
	if pr, exists := schedulingResult.ProfileResults[experimentalPrefillProfile]; exists && len(pr.TargetEndpoints) > 0 {
		prefillMeta := pr.TargetEndpoints[0].GetMetadata()
		prefillPod := kvblock.PodEntry{
			PodIdentifier: fmt.Sprintf("%s:%s", prefillMeta.Address, prefillMeta.Port),
			Speculative:   true,
		}
		for _, keys := range state.blockKeys {
			if len(keys) == 0 {
				continue
			}
			if err := index.Add(ctx, nil, keys, []kvblock.PodEntry{prefillPod}); err != nil {
				logger.Error(err, "Failed to add speculative entries for prefill endpoint",
					"pod", prefillPod.PodIdentifier)
			}
		}
		allPodEntries = append(allPodEntries, prefillPod)
	}

	// 5. Register in TTL cache for automatic eviction
	flatKeys := slices.Concat(state.blockKeys...)
	s.speculativeCache.Set(request.RequestID, &speculativeEntries{
		blockKeys:  flatKeys,
		podEntries: allPodEntries,
	}, s.speculativeTTL)

	logger.V(logging.TRACE).Info("Added speculative entries",
		"requestID", request.RequestID,
		"pod", speculativePod.PodIdentifier,
		"blockKeys", len(flatKeys),
		"ttl", s.speculativeTTL)
}

// --- EndpointExtractor implementation ---

// ExpectedInputType declares the data type this extractor consumes.
// Required by the data layer's source/extractor type-compatibility check.
func (s *Scorer) ExpectedInputType() reflect.Type {
	return fwkdl.EndpointEventReflectType
}

// ExtractEndpoint reacts to endpoint lifecycle events from the data layer's
// endpoint-notification-source: an add/update installs a per-pod ZMQ
// subscriber so KV-cache events flow into the index; a delete tears it down.
// No-op when DiscoverPods is disabled or the namespaced name is unavailable.
//
// Being called at all is also the signal that the data layer is wired for
// this scorer; the legacy in-Score discovery path turns itself off from
// here on.
func (s *Scorer) ExtractEndpoint(ctx context.Context, event fwkdl.EndpointEvent) error {
	s.extractorActive.Store(true)
	if !s.kvEventsConfig.DiscoverPods || s.kvEventsConfig.PodDiscoveryConfig == nil {
		return nil
	}
	meta := event.Endpoint.GetMetadata()
	if meta == nil || meta.NamespacedName.Name == "" {
		return nil
	}

	logger := log.FromContext(ctx).WithName(s.typedName.String())
	endpointKey := meta.NamespacedName.String()

	switch event.Type {
	case fwkdl.EventAddOrUpdate:
		if err := s.ensureSubscriber(ctx, meta); err != nil {
			return err
		}
		logger.V(logging.DEBUG).Info("Adding subscriber", "endpoint", endpointKey)
	case fwkdl.EventDelete:
		s.subscribersManager.RemoveSubscriber(ctx, endpointKey)
		logger.V(logging.DEBUG).Info("Removed KV-events subscriber", "endpoint", endpointKey)
	}
	return nil
}

// ensureSubscriber installs (or refreshes) a per-pod ZMQ subscriber for the
// given endpoint metadata. Used by both the data-layer-driven extractor path
// and the legacy in-Score backwards-compat path. Returns nil for endpoints
// without an address — those can't be dialed.
//
// The subscriber goroutine is started against subscriberCtx (plugin-lifetime),
// not the caller ctx, so request-scoped contexts don't tear it down.
func (s *Scorer) ensureSubscriber(ctx context.Context, meta *fwkdl.EndpointMetadata) error {
	if meta == nil || meta.Address == "" {
		return nil
	}
	endpointKey := meta.NamespacedName.String()
	zmqEndpoint := fmt.Sprintf("tcp://%s:%d", meta.Address, s.kvEventsConfig.PodDiscoveryConfig.SocketPort)

	logger := log.FromContext(ctx).WithName(s.typedName.String())
	if err := s.subscribersManager.EnsureSubscriber(s.subscriberCtx, endpointKey,
		zmqEndpoint, s.kvEventsConfig.TopicFilter, true); err != nil {
		logger.Error(err, "Failed to ensure KV-events subscriber for endpoint",
			"endpoint", endpointKey, "address", meta.Address)
		return fmt.Errorf("ensure subscriber for %s: %w", endpointKey, err)
	}
	logger.V(logging.DEBUG).Info("Ensured KV-events subscriber", "endpoint", endpointKey, "zmq", zmqEndpoint)
	return nil
}

// ensureSubscribersForEndpoints is the backwards-compat path for configs
// that don't wire the endpoint-notification-source: it iterates the candidate
// endpoints presented to Score and ensures a per-pod subscriber for each.
// Errors are logged and not returned — discovery is best-effort here, the
// data layer remains the authoritative source when wired.
//
// Becomes a no-op once ExtractEndpoint has been called at least once,
// indicating the data layer is wired and will drive subscriber lifecycle.
func (s *Scorer) ensureSubscribersForEndpoints(ctx context.Context, endpoints []scheduling.Endpoint) {
	if s.extractorActive.Load() {
		return
	}
	if !s.kvEventsConfig.DiscoverPods || s.kvEventsConfig.PodDiscoveryConfig == nil {
		return
	}
	for _, ep := range endpoints {
		_ = s.ensureSubscriber(ctx, ep.GetMetadata())
	}
}

// --- Internal helper methods ---

// computeBlockKeys extracts block keys from an LLM request. Prefers the
// tokens written by the token-producer DataProducer plugin; falls back to
// the deprecated prompt-string path on the indexer when no tokens are
// attached. Returns nil keys (no error) when neither path can produce them.
func (s *Scorer) computeBlockKeys(ctx context.Context,
	request *scheduling.InferenceRequest) ([]kvblock.BlockHash, error) {
	if request.Body == nil {
		return nil, nil
	}

	if tp := request.Body.TokenizedPrompt; tp != nil && len(tp.TokenIDs) > 0 {
		var extraFeatures []*kvblock.BlockExtraFeatures
		if len(tp.MultiModalFeatures) > 0 {
			mmHashes, mmPlaceholders := tokenizer.ConvertMMFeaturesFromUpstream(tp.MultiModalFeatures)
			extraFeatures = kvblock.ComputeBlockExtraFeatures(
				mmHashes, mmPlaceholders, s.blockSizeTokens, len(tp.TokenIDs))
		}
		return s.kvCacheIndexer.ComputeBlockKeysFromTokens(ctx, tp.TokenIDs, request.TargetModel, extraFeatures)
	}

	var (
		keys []kvblock.BlockHash
		err  error
	)
	switch {
	case request.Body.ChatCompletions != nil:
		renderReq := tokenizer.ChatCompletionsToRenderChatRequest(request.Body.ChatCompletions)
		//nolint:staticcheck // SA1019: legacy path retained for tokenizersPoolConfig configs.
		keys, err = s.kvCacheIndexer.ComputeBlockKeys(ctx, renderReq, "", request.TargetModel)
	case request.Body.Completions != nil:
		prompt := request.Body.Completions.Prompt
		if len(prompt.Strings) > 0 {
			var allBlockKeys []kvblock.BlockHash
			for _, promptStr := range prompt.Strings {
				//nolint:staticcheck // SA1019: legacy path retained for tokenizersPoolConfig configs.
				pKeys, pErr := s.kvCacheIndexer.ComputeBlockKeys(ctx, nil, promptStr, request.TargetModel)
				if pErr != nil {
					if errors.Is(pErr, kvcache.ErrInternalTokenizationDisabled) {
						return nil, nil
					}
					return nil, pErr
				}
				allBlockKeys = append(allBlockKeys, pKeys...)
			}
			return allBlockKeys, nil
		}
		//nolint:staticcheck // SA1019: legacy path retained for tokenizersPoolConfig configs.
		keys, err = s.kvCacheIndexer.ComputeBlockKeys(ctx, nil, prompt.Raw, request.TargetModel)
	default:
		return nil, nil
	}
	if errors.Is(err, kvcache.ErrInternalTokenizationDisabled) {
		return nil, nil
	}
	return keys, err
}

// extractPodSet builds a set of pod identifiers from endpoints for filtered index lookups.
func extractPodSet(endpoints []scheduling.Endpoint) sets.Set[string] {
	podSet := sets.New[string]()
	for _, ep := range endpoints {
		if m := ep.GetMetadata(); m != nil {
			podSet.Insert(fmt.Sprintf("%s:%s", m.Address, m.Port))
		}
	}
	return podSet
}

// aggregatePromptScores uses total cached-work semantics for multi-prompt
// completions: each prompt contributes its raw prefix-cache score to the pod.
func aggregatePromptScores(perPromptScores []map[string]float64) map[string]float64 {
	aggregated := make(map[string]float64)
	for _, scores := range perPromptScores {
		for pod, score := range scores {
			aggregated[pod] += score
		}
	}
	return aggregated
}

// getBlockSizeTokens returns the block size in tokens from the token processor config.
func (s *Scorer) getBlockSizeTokens() int {
	return s.blockSizeTokens
}

func perPromptTokenGroups(request *scheduling.InferenceRequest, tp *scheduling.TokenizedPrompt) ([][]uint32, bool, string) {
	if tp == nil || len(tp.PerPromptTokens) == 0 {
		return nil, false, ""
	}
	if request == nil || request.Body == nil || request.Body.Completions == nil {
		return nil, false, "per-prompt tokens require a completions request body"
	}
	expectedPromptCount := len(request.Body.Completions.Prompt.Strings)
	if expectedPromptCount <= 1 {
		return nil, false, "per-prompt tokens require a completions string-array prompt with more than one prompt"
	}
	if len(tp.PerPromptTokens) != expectedPromptCount {
		return nil, false, fmt.Sprintf("per-prompt token count %d does not match prompt count %d",
			len(tp.PerPromptTokens), expectedPromptCount)
	}
	if !slices.Equal(slices.Concat(tp.PerPromptTokens...), tp.TokenIDs) {
		return nil, false, "per-prompt tokens do not match flattened token IDs"
	}
	return tp.PerPromptTokens, true, ""
}

// getScores retrieves the endpoint scores from the KV-cache indexer
// based on the provided LLM request.
// If tokenized prompt data is found on the request (written by the tokenizer
// DataProducer plugin), it calls ScoreTokens directly, bypassing
// prompt/chat tokenization. Otherwise, chat completions and regular
// completions are tokenized internally by the kvcache indexer.
func (s *Scorer) getScores(ctx context.Context, _ *scheduling.CycleState, request *scheduling.InferenceRequest) (map[string]float64, error) {
	logger := log.FromContext(ctx).WithName(s.typedName.String())
	traceLogger := logger.V(logging.TRACE)

	traceLogger.Info("Getting scores",
		"isChatCompletions", request.Body != nil && request.Body.ChatCompletions != nil,
		"isCompletions", request.Body != nil && request.Body.Completions != nil)

	// Prefer pre-tokenized input from the tokenizer DataProducer plugin.
	if request.Body != nil {
		if tp := request.Body.TokenizedPrompt; tp != nil && len(tp.TokenIDs) > 0 {
			if promptTokenGroups, ok, reason := perPromptTokenGroups(request, tp); len(tp.PerPromptTokens) > 0 {
				if !ok {
					traceLogger.Info("Ignoring invalid per-prompt tokens, using flat token scoring", "reason", reason)
				} else {
					traceLogger.Info("multi-prompt tokens found on request, scoring independently",
						"promptCount", len(promptTokenGroups))

					perPromptScores := make([]map[string]float64, 0, len(promptTokenGroups))
					for _, promptTokens := range promptTokenGroups {
						if len(promptTokens) == 0 {
							continue
						}
						scores, scoreErr := s.kvCacheIndexer.ScoreTokens(ctx, promptTokens, request.TargetModel, nil, nil)
						if scoreErr != nil {
							return nil, fmt.Errorf("failed to get endpoint scores for tokens: %w", scoreErr)
						}
						perPromptScores = append(perPromptScores, scores)
					}
					return aggregatePromptScores(perPromptScores), nil
				}
			}

			// Single-prompt path: use concatenated TokenIDs directly.
			traceLogger.Info("tokens found on request, skipping tokenization")

			var extraFeatures []*kvblock.BlockExtraFeatures
			if len(tp.MultiModalFeatures) > 0 {
				mmHashes, mmPlaceholders := tokenizer.ConvertMMFeaturesFromUpstream(tp.MultiModalFeatures)
				extraFeatures = kvblock.ComputeBlockExtraFeatures(
					mmHashes, mmPlaceholders, s.blockSizeTokens, len(tp.TokenIDs))
			}

			scores, err := s.kvCacheIndexer.ScoreTokens(ctx, tp.TokenIDs, request.TargetModel, nil, extraFeatures)
			if err != nil {
				return nil, fmt.Errorf("failed to get endpoint scores for tokens: %w", err)
			}
			return scores, nil
		}
	}

	// The upstream parser guarantees exactly one body is populated, but we defensively prioritize chat completions.
	// If an unexpected dual payload slips through (parser regression/new client), log it and use chat semantics.
	if request.Body != nil && request.Body.ChatCompletions != nil {
		if request.Body.Completions != nil {
			traceLogger.Info("Both chat/completions and completions present; defaulting to chat/completions")
		}

		renderReq := tokenizer.ChatCompletionsToRenderChatRequest(request.Body.ChatCompletions)

		traceLogger.Info("Processing chat completion request",
			"messagesCount", len(renderReq.Conversation),
			"toolsCount", len(renderReq.Tools),
			"documentsCount", len(renderReq.Documents))

		//nolint:staticcheck // SA1019: legacy path retained for tokenizersPoolConfig configs.
		scores, err := s.kvCacheIndexer.GetPodScores(ctx, renderReq, "", request.TargetModel, nil)
		if err != nil {
			if errors.Is(err, kvcache.ErrInternalTokenizationDisabled) {
				return map[string]float64{}, nil
			}
			return nil, fmt.Errorf("failed to get endpoint scores for chat/completions: %w", err)
		}
		return scores, nil
	}

	// For regular completions, use the prompt directly
	if request.Body != nil && request.Body.Completions != nil {
		prompt := request.Body.Completions.Prompt
		if len(prompt.Strings) > 0 {
			perPromptScores := make([]map[string]float64, 0, len(prompt.Strings))
			for _, promptStr := range prompt.Strings {
				traceLogger.Info("Scoring completion prompt independently", "promptLength", len(promptStr))
				scores, err := s.kvCacheIndexer.GetPodScores(ctx, nil, promptStr, request.TargetModel, nil)
				if err != nil {
					return nil, fmt.Errorf("failed to get endpoint scores for completions: %w", err)
				}
				perPromptScores = append(perPromptScores, scores)
			}
			return aggregatePromptScores(perPromptScores), nil
		}

		//nolint:staticcheck // SA1019: legacy path retained for tokenizersPoolConfig configs.
		scores, err := s.kvCacheIndexer.GetPodScores(ctx, nil, prompt.Raw, request.TargetModel, nil)
		if err != nil {
			if errors.Is(err, kvcache.ErrInternalTokenizationDisabled) {
				return map[string]float64{}, nil
			}
			return nil, fmt.Errorf("failed to get endpoint scores for completions: %w", err)
		}
		return scores, nil
	}

	return nil, errors.New("no valid input found in request")
}
