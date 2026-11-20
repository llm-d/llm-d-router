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

// Package lengthadaptiveload implements the LengthAdaptiveLoadScorer plugin for llm-d-router.
// It is inspired by the Decode Load Balancer design in DeepSeek's DeepInfer
// (https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md),
// dynamically blending request-concurrency scoring and token-load scoring based on
// token-producer's estimate of the incoming request length.
package lengthadaptiveload

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"sync"
	"sync/atomic"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/tokenizer"
)

const (
	// LengthAdaptiveLoadScorerType is the registered plugin type name.
	LengthAdaptiveLoadScorerType = "length-adaptive-load-scorer"

	// DefaultMaxConcurrencyLabel is the default pod label key used to read per-endpoint
	// maximum concurrent requests (e.g., vLLM max_num_seqs) in heterogeneous pools.
	DefaultMaxConcurrencyLabel = "llm-d.ai/max-num-seqs"

	// DefaultMaxTokenCapacityLabel is the default pod label key used to read per-endpoint
	// KV cache token capacity in heterogeneous pools.
	DefaultMaxTokenCapacityLabel = "llm-d.ai/max-kv-tokens" //#nosec G101 -- pod label key, not a credential

	defaultShortThresholdTokens = 512
	defaultLongThresholdTokens  = 16384
	defaultMaxConcurrency       = 256
	defaultMaxTokenCapacity     = 150000
)

// Parameters holds the configuration for LengthAdaptiveLoadScorer.
type Parameters struct {
	// ShortThresholdTokens defines the prompt token count at or below which
	// the scorer routes purely by active request count (alpha = 0.0).
	// Default: 512
	ShortThresholdTokens int `json:"shortThresholdTokens,omitempty"`

	// LongThresholdTokens defines the prompt token count at or above which
	// the scorer routes purely by in-flight token load (alpha = 1.0).
	// Default: 16384
	LongThresholdTokens int `json:"longThresholdTokens,omitempty"`

	// MaxConcurrency defines the default maximum concurrent requests per endpoint (e.g., vLLM --max-num-seqs).
	// Default: 256
	MaxConcurrency int `json:"maxConcurrency,omitempty"`

	// MaxTokenCapacity defines the default maximum KV cache token capacity per endpoint.
	// Default: 150000
	MaxTokenCapacity int `json:"maxTokenCapacity,omitempty"`

	// MaxConcurrencyLabel is the pod label key checked first for per-endpoint max concurrency
	// before falling back to MaxConcurrency.
	// Default: "llm-d.ai/max-num-seqs"
	MaxConcurrencyLabel string `json:"maxConcurrencyLabel,omitempty"`

	// MaxTokenCapacityLabel is the pod label key checked first for per-endpoint KV token capacity
	// before falling back to MaxTokenCapacity.
	// Default: "llm-d.ai/max-kv-tokens"
	MaxTokenCapacityLabel string `json:"maxTokenCapacityLabel,omitempty"`

	// InFlightLoadProducerName optionally specifies a named inflight-load-producer instance.
	InFlightLoadProducerName string `json:"inFlightLoadProducerName,omitempty"`
}

// compile-time type assertions
var (
	_ fwksched.Scorer             = &LengthAdaptiveLoadScorer{}
	_ fwkplugin.ConsumerPlugin    = &LengthAdaptiveLoadScorer{}
	_ fwkrh.PreRequest            = &LengthAdaptiveLoadScorer{}
	_ fwkrh.ResponseBodyProcessor = &LengthAdaptiveLoadScorer{}
)

// decodeTokenEntry tracks a request's prompt KV token footprint on a decode endpoint
// for the entire duration of the request (until EndOfStream or stream cancellation),
// preventing early token release at StartOfStream when prefill finishes.
type decodeTokenEntry struct {
	tokens  atomic.Int64
	counter *atomic.Int64
}

var _ fwkplugin.EvictableStateData = (*decodeTokenEntry)(nil)

func (e *decodeTokenEntry) Clone() fwkplugin.StateData {
	if e == nil {
		return nil
	}
	clone := &decodeTokenEntry{counter: e.counter}
	clone.tokens.Store(e.tokens.Load())
	return clone
}

func (e *decodeTokenEntry) OnEvicted(_ string, _ fwkplugin.StateKey) {
	if t := e.tokens.Swap(0); t != 0 && e.counter != nil {
		for {
			cur := e.counter.Load()
			next := max(cur-t, 0)
			if e.counter.CompareAndSwap(cur, next) {
				break
			}
		}
	}
}

// LengthAdaptiveLoadScorer scores candidate endpoints by dynamically blending
// request-concurrency headroom and token-load headroom based on incoming request length.
type LengthAdaptiveLoadScorer struct {
	typedName           fwkplugin.TypedName
	params              Parameters
	inFlightLoadDataKey fwkplugin.DataKey
	pluginState         *fwkplugin.PluginState
	decodeTokensMu      sync.RWMutex
	decodeTokens        map[string]*atomic.Int64
}

// Factory constructs a LengthAdaptiveLoadScorer instance from JSON parameters.
func Factory(name string, rawParameters *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := Parameters{
		ShortThresholdTokens:  defaultShortThresholdTokens,
		LongThresholdTokens:   defaultLongThresholdTokens,
		MaxConcurrency:        defaultMaxConcurrency,
		MaxTokenCapacity:      defaultMaxTokenCapacity,
		MaxConcurrencyLabel:   DefaultMaxConcurrencyLabel,
		MaxTokenCapacityLabel: DefaultMaxTokenCapacityLabel,
	}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse parameters of %q scorer: %w", LengthAdaptiveLoadScorerType, err)
		}
	}
	ctx := context.Background()
	if handle != nil && handle.Context() != nil {
		ctx = handle.Context()
	}
	scorer := NewLengthAdaptiveLoadScorer(&params).WithName(name)
	scorer.pluginState = fwkplugin.NewPluginState(ctx)
	return scorer, nil
}

// NewLengthAdaptiveLoadScorer initializes a new LengthAdaptiveLoadScorer with validated parameters.
func NewLengthAdaptiveLoadScorer(params *Parameters) *LengthAdaptiveLoadScorer {
	cfg := Parameters{
		ShortThresholdTokens:  defaultShortThresholdTokens,
		LongThresholdTokens:   defaultLongThresholdTokens,
		MaxConcurrency:        defaultMaxConcurrency,
		MaxTokenCapacity:      defaultMaxTokenCapacity,
		MaxConcurrencyLabel:   DefaultMaxConcurrencyLabel,
		MaxTokenCapacityLabel: DefaultMaxTokenCapacityLabel,
	}
	if params != nil {
		if params.ShortThresholdTokens > 0 {
			cfg.ShortThresholdTokens = params.ShortThresholdTokens
		}
		if params.LongThresholdTokens > 0 {
			cfg.LongThresholdTokens = params.LongThresholdTokens
		}
		if params.MaxConcurrency > 0 {
			cfg.MaxConcurrency = params.MaxConcurrency
		}
		if params.MaxTokenCapacity > 0 {
			cfg.MaxTokenCapacity = params.MaxTokenCapacity
		}
		if params.MaxConcurrencyLabel != "" {
			cfg.MaxConcurrencyLabel = params.MaxConcurrencyLabel
		}
		if params.MaxTokenCapacityLabel != "" {
			cfg.MaxTokenCapacityLabel = params.MaxTokenCapacityLabel
		}
		cfg.InFlightLoadProducerName = params.InFlightLoadProducerName
	}
	if cfg.LongThresholdTokens <= cfg.ShortThresholdTokens {
		cfg.LongThresholdTokens = cfg.ShortThresholdTokens + 1
	}

	return &LengthAdaptiveLoadScorer{
		typedName:           fwkplugin.TypedName{Type: LengthAdaptiveLoadScorerType, Name: LengthAdaptiveLoadScorerType},
		params:              cfg,
		inFlightLoadDataKey: attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(cfg.InFlightLoadProducerName),
		pluginState:         fwkplugin.NewPluginState(context.Background()),
		decodeTokens:        make(map[string]*atomic.Int64),
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (s *LengthAdaptiveLoadScorer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// WithName sets the instance name of the scorer.
func (s *LengthAdaptiveLoadScorer) WithName(name string) *LengthAdaptiveLoadScorer {
	if name != "" {
		s.typedName.Name = name
	}
	return s
}

// Category returns the preference the scorer applies when scoring candidate endpoints.
func (s *LengthAdaptiveLoadScorer) Category() fwksched.ScorerCategory {
	return fwksched.Distribution
}

// Consumes declares the request and endpoint data keys consumed by this scorer.
func (s *LengthAdaptiveLoadScorer) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{
		Required: map[fwkplugin.DataKey]any{
			tokenizer.TokenizedPromptDataKey: fwkrequest.TokenizedRequest{},
			s.inFlightLoadDataKey:            attrconcurrency.InFlightLoad{},
		},
	}
}

// computeAlpha computes the length-adaptive weight alpha(L_r) in [0.0, 1.0].
func (s *LengthAdaptiveLoadScorer) computeAlpha(reqTokens int) float64 {
	if reqTokens <= s.params.ShortThresholdTokens {
		return 0.0
	}
	if reqTokens >= s.params.LongThresholdTokens {
		return 1.0
	}
	span := float64(s.params.LongThresholdTokens - s.params.ShortThresholdTokens)
	return float64(reqTokens-s.params.ShortThresholdTokens) / span
}

// estimateRequestTokens extracts the new incoming request's token count from token-producer
// (with a fallback to raw request byte length / BytesPerToken).
func estimateRequestTokens(request *fwksched.InferenceRequest) int {
	// 1. Primary: token-producer output on the request body.
	if request != nil && request.Body != nil && request.Body.TokenizedRequest != nil {
		if n := request.Body.TokenizedRequest.TokenCount(); n > 0 {
			return n
		}
	}
	// 2. Fallback: byte-length heuristic when token-producer is not active.
	if request != nil && request.RequestSizeBytes > 0 {
		return request.RequestSizeBytes / fwkrequest.BytesPerToken
	}
	return 0
}

type inflightPodLoad struct {
	Requests int
	Tokens   int
}

func (s *LengthAdaptiveLoadScorer) getOrCreateDecodeCounter(endpointID string) *atomic.Int64 {
	s.decodeTokensMu.RLock()
	c, ok := s.decodeTokens[endpointID]
	s.decodeTokensMu.RUnlock()
	if ok {
		return c
	}
	s.decodeTokensMu.Lock()
	defer s.decodeTokensMu.Unlock()
	if c, ok = s.decodeTokens[endpointID]; ok {
		return c
	}
	c = &atomic.Int64{}
	s.decodeTokens[endpointID] = c
	return c
}

func (s *LengthAdaptiveLoadScorer) getDecodeTokens(endpointID string) int {
	if endpointID == "" {
		return 0
	}
	s.decodeTokensMu.RLock()
	c, ok := s.decodeTokens[endpointID]
	s.decodeTokensMu.RUnlock()
	if !ok || c == nil {
		return 0
	}
	return int(max(c.Load(), 0))
}

// PreRequest records the scheduled request's full prompt KV token footprint on the
// primary (decode) endpoint for the entire lifetime of the request until EndOfStream.
func (s *LengthAdaptiveLoadScorer) PreRequest(
	ctx context.Context,
	request *fwksched.InferenceRequest,
	result *fwksched.SchedulingResult,
) error {
	if result == nil || request == nil || request.RequestID == "" || s.pluginState == nil {
		return nil
	}
	endpoint := result.PrimaryEndpoint()
	if endpoint == nil || endpoint.GetMetadata() == nil {
		return nil
	}
	eid := endpoint.GetMetadata().ID.String()
	if eid == "" {
		return nil
	}
	reqTokens := int64(estimateRequestTokens(request))
	if reqTokens <= 0 {
		return nil
	}
	counter := s.getOrCreateDecodeCounter(eid)
	counter.Add(reqTokens)
	entry := &decodeTokenEntry{counter: counter}
	entry.tokens.Store(reqTokens)
	s.pluginState.Write(request.RequestID, fwkplugin.StateKey(eid+"|"+result.PrimaryProfileName+"|decode_tokens"), entry)
	s.pluginState.BindLiveness(ctx, request.RequestID)
	return nil
}

// ResponseBody keeps the decode KV token footprint alive during streaming and
// releases it only at EndOfStream.
func (s *LengthAdaptiveLoadScorer) ResponseBody(
	_ context.Context,
	request *fwksched.InferenceRequest,
	resp *fwkrh.Response,
	_ *datalayer.EndpointMetadata,
) {
	if request == nil || resp == nil || request.RequestID == "" || s.pluginState == nil {
		return
	}
	if resp.EndOfStream {
		s.pluginState.Delete(request.RequestID)
	} else {
		s.pluginState.Touch(request.RequestID)
	}
}

// readInflightPodLoad reads the endpoint's current in-flight requests and tokens
// tracked by inflight-load-producer, taking the maximum with the scorer's own
// decode-lifetime token counter so tokens remain accounted for between StartOfStream
// and EndOfStream.
func (s *LengthAdaptiveLoadScorer) readInflightPodLoad(endpoint fwksched.Endpoint) inflightPodLoad {
	var load inflightPodLoad
	if endpoint == nil {
		return load
	}
	if val, ok := endpoint.Get(s.inFlightLoadDataKey); ok {
		if fl, ok := val.(*attrconcurrency.InFlightLoad); ok && fl != nil {
			load.Requests = int(fl.Requests)
			load.Tokens = int(fl.Tokens)
		}
	}
	if md := endpoint.GetMetadata(); md != nil {
		if tracked := s.getDecodeTokens(md.ID.String()); tracked > load.Tokens {
			load.Tokens = tracked
		}
	}
	return load
}

type endpointCapacity struct {
	MaxConcurrency   int
	MaxTokenCapacity int
}

// resolveEndpointCapacity performs a 2-tier lookup for an endpoint's concurrency and KV token
// capacity: it checks per-endpoint metadata labels first (for heterogeneous accelerator pools)
// and falls back to the plugin's MaxConcurrency and MaxTokenCapacity parameters.
func (s *LengthAdaptiveLoadScorer) resolveEndpointCapacity(endpoint fwksched.Endpoint) endpointCapacity {
	podCap := endpointCapacity{
		MaxConcurrency:   s.params.MaxConcurrency,
		MaxTokenCapacity: s.params.MaxTokenCapacity,
	}
	if endpoint == nil {
		return podCap
	}
	md := endpoint.GetMetadata()
	if md == nil || len(md.Labels) == 0 {
		return podCap
	}
	if s.params.MaxConcurrencyLabel != "" {
		if raw, ok := md.Labels[s.params.MaxConcurrencyLabel]; ok {
			if v, err := strconv.Atoi(raw); err == nil && v > 0 {
				podCap.MaxConcurrency = v
			}
		}
	}
	if s.params.MaxTokenCapacityLabel != "" {
		if raw, ok := md.Labels[s.params.MaxTokenCapacityLabel]; ok {
			if v, err := strconv.Atoi(raw); err == nil && v > 0 {
				podCap.MaxTokenCapacity = v
			}
		}
	}
	return podCap
}

// Score scores candidate endpoints in [0.0, 1.0] using length-adaptive blending of
// request concurrency and token load, guarded by per-endpoint capacity ceilings.
func (s *LengthAdaptiveLoadScorer) Score(
	ctx context.Context,
	request *fwksched.InferenceRequest,
	endpoints []fwksched.Endpoint,
) map[fwksched.Endpoint]float64 {
	scores := make(map[fwksched.Endpoint]float64, len(endpoints))
	if len(endpoints) == 0 {
		return scores
	}

	reqTokens := estimateRequestTokens(request)
	alpha := s.computeAlpha(reqTokens)

	maxReqRatio := 1.0
	maxTokRatio := 1.0
	podLoads := make(map[fwksched.Endpoint]inflightPodLoad, len(endpoints))
	podCaps := make(map[fwksched.Endpoint]endpointCapacity, len(endpoints))

	for _, endpoint := range endpoints {
		load := s.readInflightPodLoad(endpoint)
		podCap := s.resolveEndpointCapacity(endpoint)
		podLoads[endpoint] = load
		podCaps[endpoint] = podCap
		if reqRatio := float64(load.Requests) / float64(podCap.MaxConcurrency); reqRatio > maxReqRatio {
			maxReqRatio = reqRatio
		}
		if tokRatio := float64(load.Tokens) / float64(podCap.MaxTokenCapacity); tokRatio > maxTokRatio {
			maxTokRatio = tokRatio
		}
	}

	debugLogger := log.FromContext(ctx).V(logging.DEBUG)
	debugEnabled := debugLogger.Enabled()

	anyWithinGuardrail := false
	rawScores := make(map[fwksched.Endpoint]float64, len(endpoints))

	for _, endpoint := range endpoints {
		load := podLoads[endpoint]
		podCap := podCaps[endpoint]
		reqRatio := float64(load.Requests) / float64(podCap.MaxConcurrency)
		tokRatio := float64(load.Tokens) / float64(podCap.MaxTokenCapacity)

		queueScore := 1.0 - math.Min(1.0, reqRatio/maxReqRatio)
		tokenScore := 1.0 - math.Min(1.0, tokRatio/maxTokRatio)

		blended := (1.0-alpha)*queueScore + alpha*tokenScore
		rawScores[endpoint] = blended

		if (load.Requests+1) <= podCap.MaxConcurrency && (load.Tokens+reqTokens) <= podCap.MaxTokenCapacity {
			scores[endpoint] = blended
			anyWithinGuardrail = true
		} else {
			scores[endpoint] = 0.0
		}

		if debugEnabled {
			endpointID := ""
			if md := endpoint.GetMetadata(); md != nil {
				endpointID = md.ID.String()
			}
			debugLogger.Info("LengthAdaptiveLoadScorer scoring",
				"endpoint", endpointID,
				"reqTokens", reqTokens,
				"alpha", alpha,
				"inFlightRequests", load.Requests,
				"inFlightTokens", load.Tokens,
				"maxConcurrency", podCap.MaxConcurrency,
				"maxTokenCapacity", podCap.MaxTokenCapacity,
				"queueScore", queueScore,
				"tokenScore", tokenScore,
				"score", scores[endpoint],
			)
		}
	}

	if !anyWithinGuardrail {
		return rawScores
	}
	return scores
}
