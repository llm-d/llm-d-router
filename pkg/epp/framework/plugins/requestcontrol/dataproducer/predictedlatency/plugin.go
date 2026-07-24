/*
Copyright 2025 The Kubernetes Authors.

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

package predictedlatency

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/jellydator/ttlcache/v3"
	latencypredictor "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/predictedlatency/latencypredictorclient"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	latencyproducerconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/predictedlatency/constants"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

const (
	// LatencyDataProviderPluginType is the plugin type for the latency predictor.
	// It trains XGBoost models via the sidecar and generates predictions for scoring.
	LatencyDataProviderPluginType = latencyproducerconstants.LatencyDataProviderPluginType

	// TTFTSLOHeaderKey is the header key for the TTFT SLO.
	TTFTSLOHeaderKey = metadata.TTFTSLOHeaderKey
	// TPOTSLOHeaderKey is the header key for the TPOT SLO.
	TPOTSLOHeaderKey = metadata.TPOTSLOHeaderKey

	// ExperimentalDefaultPrefillProfile is the default profile name for prefill endpoints in disaggregated serving.
	ExperimentalDefaultPrefillProfile = "prefill"
)

// PredictedLatency is the latency data provider plugin. It handles:
//   - Produce: bulk predictions via the latency predictor sidecar
//   - PreRequest: dispatch-time bookkeeping (token counters, request queues)
//   - ResponseHeader/ResponseBody: training data collection (TTFT/TPOT)
//   - Produces/Consumes: endpoint attribute declarations
//
// Scoring, picking, and admission are handled by separate sub-plugins:
// LatencyScorer, AffinityWeightedPicker, and LatencyAdmission.
type PredictedLatency struct {
	typedName                    plugin.TypedName
	latencypredictor             latencypredictor.PredictorInterface
	runningRequestLists          sync.Map                                      // Key: types.NamespacedName, Value: *requestPriorityQueue
	sloContextStore              *ttlcache.Cache[string, *predictedLatencyCtx] // TTL cache for request contexts
	config                       Config
	prefixMatchDataKey           plugin.DataKey
	inFlightLoadDataKey          plugin.DataKey
	latencyPredictionInfoDataKey plugin.DataKey
}

// endpointInFlightLoad reads the InFlightLoad attribute published by the
// configured InFlightLoadProducer for the endpoint. The producer discounts the
// already-cached prompt prefix, so Tokens reflects the uncached prefill work in
// flight and Requests the active request count. Returns false when the
// attribute is absent (e.g. an endpoint added before the producer injected it).
func (pl *PredictedLatency) endpointInFlightLoad(endpoint fwksched.Endpoint) (*attrconcurrency.InFlightLoad, bool) {
	if raw, ok := endpoint.Get(pl.inFlightLoadDataKey.String()); ok {
		if load, ok := raw.(*attrconcurrency.InFlightLoad); ok && load != nil {
			return load, true
		}
	}
	return nil, false
}

const maxDebugDumpEndpoints = 100

var _ plugin.StateDumper = &PredictedLatency{}

type predictedLatencyState struct {
	Endpoints       []endpointPredictedLatencyState `json:"endpoints"`
	TotalEndpoints  int                             `json:"totalEndpoints"`
	MaxEndpoints    int                             `json:"maxEndpoints"`
	Truncated       bool                            `json:"truncated"`
	TrackedRequests int                             `json:"trackedRequests"`
}

type endpointPredictedLatencyState struct {
	Endpoint        string  `json:"endpoint"`
	RunningRequests int     `json:"runningRequests"`
	MinTPOTSLO      float64 `json:"minTpotSlo"`
}

// DumpState implements [plugin.StateDumper] and exposes per-endpoint running-request
// counts and the tightest TPOT SLO among them for the /debug/plugins/state endpoint.
//
// The prefill-tokens-in-flight signal is no longer maintained internally (it is read
// from the InFlightLoadProducer's per-endpoint attribute at scheduling time), so it is
// not reported here.
//
// The running-request queues and the context store are read under their own separate
// synchronization, so per-endpoint values are not guaranteed to be from a single
// instant. This is acceptable for a debug endpoint, where best-effort visibility is
// preferred over a global lock contending the hot path.
//
// Per-request contexts hold request payloads and are high-cardinality, so only their
// count is reported. The endpoint list is capped to the busiest endpoints.
func (pl *PredictedLatency) DumpState() (json.RawMessage, error) {
	return json.Marshal(pl.snapshotState())
}

func (pl *PredictedLatency) snapshotState() predictedLatencyState {
	type agg struct {
		running int
		minTPOT float64
	}
	endpoints := map[string]*agg{}
	get := func(id string) *agg {
		a, ok := endpoints[id]
		if !ok {
			a = &agg{}
			endpoints[id] = a
		}
		return a
	}

	pl.runningRequestLists.Range(func(key, value any) bool {
		name, ok := key.(types.NamespacedName)
		if !ok {
			return true
		}
		q, ok := value.(*requestPriorityQueue)
		if !ok || q == nil {
			return true
		}
		a := get(name.String())
		a.running = q.GetSize()
		if minReq := q.Peek(); minReq != nil {
			// Client headers can yield NaN/+Inf TPOT (strconv.ParseFloat accepts
			// them and Add only rejects negatives). json.Marshal errors on
			// non-finite floats, which would fail the whole debug response, so
			// coerce them to 0.
			if tpot := minReq.tpot; !math.IsNaN(tpot) && !math.IsInf(tpot, 0) {
				a.minTPOT = tpot
			}
		}
		return true
	})

	trackedRequests := 0
	if pl.sloContextStore != nil {
		trackedRequests = pl.sloContextStore.Len()
	}

	state := predictedLatencyState{
		Endpoints:       make([]endpointPredictedLatencyState, 0, len(endpoints)),
		TotalEndpoints:  len(endpoints),
		MaxEndpoints:    maxDebugDumpEndpoints,
		TrackedRequests: trackedRequests,
	}
	for id, a := range endpoints {
		state.Endpoints = append(state.Endpoints, endpointPredictedLatencyState{
			Endpoint:        id,
			RunningRequests: a.running,
			MinTPOTSLO:      a.minTPOT,
		})
	}

	sort.SliceStable(state.Endpoints, func(i, j int) bool {
		if state.Endpoints[i].RunningRequests != state.Endpoints[j].RunningRequests {
			return state.Endpoints[i].RunningRequests > state.Endpoints[j].RunningRequests
		}
		return state.Endpoints[i].Endpoint < state.Endpoints[j].Endpoint
	})
	if len(state.Endpoints) > maxDebugDumpEndpoints {
		state.Endpoints = state.Endpoints[:maxDebugDumpEndpoints]
		state.Truncated = true
	}

	return state
}

// inFlightLoadSnapshot is an endpoint's in-flight load captured at a single
// point in time.
type inFlightLoadSnapshot struct {
	tokens   int64
	requests int
}

// readInFlightLoad reads the endpoint's in-flight token and request load. When
// the InFlightLoad attribute is absent, tokens is zero (no in-flight token
// signal available) and the request count falls back to the endpoint's vLLM
// metrics. Both fields are always set, so the result is deterministic.
//
// The InFlightLoad attribute is a live view of the producer's tracker, not a
// stored value: the producer adds a request's own tokens in its PreRequest hook,
// and PreRequest hooks are not ordered relative to one another. This is
// therefore called only from Produce, which the data-layer DAG does order, and
// the captured value is reused for the rest of the request.
func (pl *PredictedLatency) readInFlightLoad(endpoint fwksched.Endpoint) inFlightLoadSnapshot {
	if load, ok := pl.endpointInFlightLoad(endpoint); ok {
		return inFlightLoadSnapshot{tokens: load.Tokens, requests: int(load.Requests)}
	}
	return inFlightLoadSnapshot{tokens: 0, requests: endpoint.GetMetrics().RunningRequestsSize}
}

type Config struct {
	SamplingMean                       float64       `json:"samplingMean,omitempty"`
	MaxDecodeTokenSamplesForPrediction int           `json:"maxDecodeTokenSamplesForPrediction,omitempty"`
	SLOBufferFactor                    float64       `json:"sloBufferFactor,omitempty"`
	ContextTTL                         time.Duration `json:"contextTTL,omitempty"`
	StreamingMode                      bool          `json:"streamingMode,omitempty"`
	EndpointRoleLabel                  string        `json:"endpointRoleLabel,omitempty"`
	// PredictInProduce controls whether bulk predictions are generated during
	// Produce. Set to false to disable predictions (training-only mode).
	// When false, the predictor still collects training data but does not call the
	// sidecar for predictions. Default: true.
	PredictInProduce            bool   `json:"predictInProduce,omitempty"`
	PrefixMatchInfoProducerName string `json:"prefixMatchInfoProducerName,omitempty"`
	// InFlightLoadProducerName selects which InFlightLoadProducer's per-endpoint
	// load to read for the prefill-tokens-in-flight and active-request-count
	// features. Empty defaults to the auto-created producer.
	InFlightLoadProducerName string `json:"inFlightLoadProducerName,omitempty"`
}

var DefaultConfig = Config{
	SamplingMean:                       1000,
	MaxDecodeTokenSamplesForPrediction: 0,
	SLOBufferFactor:                    1,
	ContextTTL:                         5 * time.Minute,
	StreamingMode:                      false,
	PredictInProduce:                   true,
}

func PredictedLatencyFactory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := DefaultConfig
	if rawParameters != nil {
		if err := rawParameters.Decode(&parameters); err != nil {
			return nil, fmt.Errorf("failed to unmarshal config for PredictedLatency: %w", err)
		}
	}

	if err := parameters.validate(); err != nil {
		return nil, fmt.Errorf("invalid PredictedLatency config: %w", err)
	}

	if handle == nil {
		return nil, errors.New("plugin handle is required")
	}
	if err := registerMetrics(handle.Metrics()); err != nil {
		return nil, err
	}

	predictor, err := startPredictor(handle)
	if err != nil {
		return nil, fmt.Errorf("failed to start latency predictor: %w", err)
	}

	return NewPredictedLatency(name, parameters, predictor), nil
}

func (c *Config) validate() error {
	var errs []error

	if c.SamplingMean <= 0 {
		errs = append(errs, fmt.Errorf("samplingMean must be > 0, got %f", c.SamplingMean))
	}

	if c.MaxDecodeTokenSamplesForPrediction < 0 {
		errs = append(errs, fmt.Errorf("maxDecodeTokenSamplesForPrediction must be >= 0, got %d", c.MaxDecodeTokenSamplesForPrediction))
	}

	if c.SLOBufferFactor <= 0 {
		errs = append(errs, fmt.Errorf("sloBufferFactor must be > 0, got %f", c.SLOBufferFactor))
	}

	if len(errs) > 0 {
		return errors.Join(errs...)
	}
	return nil
}

func NewPredictedLatency(name string, config Config, predictor latencypredictor.PredictorInterface) *PredictedLatency {
	predictedLatency := &PredictedLatency{
		typedName:                    plugin.TypedName{Type: LatencyDataProviderPluginType, Name: name},
		latencypredictor:             predictor,
		config:                       config,
		prefixMatchDataKey:           attrprefix.PrefixCacheMatchInfoDataKey.WithNonEmptyProducerName(config.PrefixMatchInfoProducerName),
		inFlightLoadDataKey:          attrconcurrency.InFlightLoadDataKey.WithNonEmptyProducerName(config.InFlightLoadProducerName),
		latencyPredictionInfoDataKey: attrlatency.LatencyPredictionInfoDataKey.WithNonEmptyProducerName(name),
	}

	predictedLatency.sloContextStore = ttlcache.New(
		ttlcache.WithTTL[string, *predictedLatencyCtx](config.ContextTTL),
	)

	predictedLatency.sloContextStore.OnEviction(func(ctx context.Context, reason ttlcache.EvictionReason, item *ttlcache.Item[string, *predictedLatencyCtx]) {
		if reason != ttlcache.EvictionReasonExpired {
			return
		}
		plCtx := item.Value()
		predictedLatency.removeRequestFromQueue(item.Key(), plCtx)
	})

	go predictedLatency.sloContextStore.Start()
	return predictedLatency
}

func startPredictor(handle plugin.Handle) (latencypredictor.PredictorInterface, error) {
	predictor := latencypredictor.New(latencypredictor.ConfigFromEnv(), ctrl.Log.WithName("latency-predictor-producer"))
	if err := predictor.Start(handle.Context()); err != nil {
		return nil, fmt.Errorf("failed to start latency predictor: %w", err)
	}

	go func() {
		<-handle.Context().Done()
		stopCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		predictor.Stop(stopCtx)
	}()
	return predictor, nil
}

func (pl *PredictedLatency) TypedName() plugin.TypedName {
	return pl.typedName
}

func (pl *PredictedLatency) getOrMakePredictedLatencyContextForRequest(request *fwksched.InferenceRequest) *predictedLatencyCtx {
	sloCtx, err := pl.getPredictedLatencyContextForRequest(request)
	if err != nil {
		sloCtx = newPredictedLatencyContext(request)
	}
	return sloCtx
}

// --- Per-request context ---

// predictedLatencyCtx holds per-request state for latency prediction and training.
type predictedLatencyCtx struct {
	schedulingRequest         fwksched.InferenceRequest
	targetMetadata            *fwkdl.EndpointMetadata
	prefillTargetMetadata     *fwkdl.EndpointMetadata
	schedulingResult          *fwksched.SchedulingResult
	lastSeenMetrics           map[string]*fwkdl.Metrics
	lastTokenTimestamp        time.Time
	requestReceivedTimestamp  time.Time
	generatedTokenCount       int
	incomingModelName         string
	ttft                      float64
	predictedTTFT             float64
	avgTPOT                   float64
	avgPredictedTPOT          float64
	decodeTokenSampler        *decodeTokenSampler
	tpotObservations          []float64
	predictedTPOTObservations []float64

	inputTokenCount int

	prefixCacheScoresForEndpoints map[string]float64

	// inFlightLoadForEndpoints holds the in-flight load captured for every
	// candidate endpoint during Produce, keyed by NamespacedName.String().
	// Produce is DAG-ordered, so capturing here (rather than re-reading the live
	// attribute in PreRequest, whose hook order is undefined) keeps the dispatch
	// training features deterministic and identical to the prediction features.
	inFlightLoadForEndpoints map[string]inFlightLoadSnapshot

	ttftSLO    float64
	avgTPOTSLO float64

	predictionsForScheduling map[string]endpointPredictionResult

	prefillTokensAtDispatch          int64
	prefillTokensAtDispatchOnPrefill int64
	decodeTokensAtDispatch           int64

	requestsAtDispatch          int
	requestsAtDispatchOnPrefill int
}

func newPredictedLatencyContext(request *fwksched.InferenceRequest) *predictedLatencyCtx {
	inputTokenCount := 0
	if request.Body != nil {
		if tp := request.Body.TokenizedPrompt; tp != nil {
			inputTokenCount = tp.TokenCount()
		}
	}
	return &predictedLatencyCtx{
		schedulingRequest:             *request,
		inputTokenCount:               inputTokenCount,
		lastSeenMetrics:               make(map[string]*fwkdl.Metrics),
		prefixCacheScoresForEndpoints: make(map[string]float64),
		inFlightLoadForEndpoints:      make(map[string]inFlightLoadSnapshot),
		predictionsForScheduling:      make(map[string]endpointPredictionResult),
	}
}

func (pl *PredictedLatency) getPredictedLatencyContextForRequest(request *fwksched.InferenceRequest) (*predictedLatencyCtx, error) {
	id := request.Headers[reqcommon.RequestIDHeaderKey]
	if item := pl.sloContextStore.Get(id); item != nil {
		return item.Value(), nil
	}
	return nil, fmt.Errorf("SLO context not found for request ID: %s", id)
}

func (pl *PredictedLatency) setPredictedLatencyContextForRequest(request *fwksched.InferenceRequest, ctx *predictedLatencyCtx) {
	id := request.Headers[reqcommon.RequestIDHeaderKey]
	pl.sloContextStore.Set(id, ctx, ttlcache.DefaultTTL)
}

func (pl *PredictedLatency) deletePredictedLatencyContextForRequest(request *fwksched.InferenceRequest) {
	id := request.Headers[reqcommon.RequestIDHeaderKey]
	pl.sloContextStore.Delete(id)
}

// --- Header parsing ---

// parseFloatHeader retrieves a header by name, parses it as a float64,
// and returns the value or an error if the header is missing or invalid.
func parseFloatHeader(request fwksched.InferenceRequest, headerName string) (float64, error) {
	headerValue, ok := metadata.GetLowerCaseHeaderValue(request.Headers, headerName)
	if !ok {
		return 0, nil
	}
	parsedFloat, err := strconv.ParseFloat(headerValue, 64)
	if err != nil {
		return 0, errcommon.Error{
			Code: errcommon.BadRequest,
			Msg:  headerName + " must be a float",
		}
	}
	return parsedFloat, nil
}

func (pl *PredictedLatency) parseSLOHeaders(ctx context.Context, request *fwksched.InferenceRequest, predictedLatencyCtx *predictedLatencyCtx) {
	logger := log.FromContext(ctx)
	var err error

	predictedLatencyCtx.ttftSLO, err = parseFloatHeader(*request, TTFTSLOHeaderKey)
	if err != nil {
		logger.V(logutil.DEBUG).Error(errcommon.Error{Code: errcommon.BadRequest, Msg: fmt.Sprintf("%v must be a float: %v", TTFTSLOHeaderKey, err)}, "PredictedLatency: Error parsing TTFT SLO from header")
	}

	predictedLatencyCtx.avgTPOTSLO, err = parseFloatHeader(*request, TPOTSLOHeaderKey)
	if err != nil {
		logger.V(logutil.DEBUG).Error(errcommon.Error{Code: errcommon.BadRequest, Msg: fmt.Sprintf("%v must be a float: %v", TPOTSLOHeaderKey, err)}, "PredictedLatency: Error parsing TPOT SLO from header")
	}
}

// --- Running request queue helpers ---

func (pl *PredictedLatency) getEndpointMinTPOTSLO(endpoint fwksched.Endpoint) float64 {
	endpointName := endpoint.GetMetadata().NamespacedName
	if runningReqs := pl.getRunningRequestList(endpointName); runningReqs != nil && runningReqs.GetSize() > 0 {
		if min := runningReqs.Peek(); min != nil {
			return min.tpot
		}
	}
	return 0
}

func (pl *PredictedLatency) getEndpointRunningRequestCount(endpoint fwksched.Endpoint) int {
	endpointName := endpoint.GetMetadata().NamespacedName
	if runningReqs := pl.getRunningRequestList(endpointName); runningReqs != nil {
		return runningReqs.GetSize()
	}
	return 0
}

func (pl *PredictedLatency) getRunningRequestList(endpointName types.NamespacedName) *requestPriorityQueue {
	if value, ok := pl.runningRequestLists.Load(endpointName); ok {
		return value.(*requestPriorityQueue)
	}
	return nil
}

func (pl *PredictedLatency) removeRequestFromEndpoint(endpointName types.NamespacedName, requestID string) {
	if queue := pl.getRunningRequestList(endpointName); queue != nil {
		queue.Remove(requestID)
		if queue.GetSize() == 0 {
			pl.runningRequestLists.Delete(endpointName)
		}
	}
}

func (pl *PredictedLatency) removeRequestFromQueue(requestID string, ctx *predictedLatencyCtx) {
	if ctx == nil || ctx.targetMetadata == nil {
		return
	}
	endpointName := types.NamespacedName{
		Name:      ctx.targetMetadata.NamespacedName.Name,
		Namespace: ctx.targetMetadata.NamespacedName.Namespace,
	}
	pl.removeRequestFromEndpoint(endpointName, requestID)
}
