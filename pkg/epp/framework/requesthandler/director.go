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

// Package requesthandler defines the Director component responsible for orchestrating request processing after initial
// parsing.
package requesthandler

import (
	"context"
	"encoding/json"
	"fmt"
	"math/rand"
	"net"
	"strings"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/requesthandler/types"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	// TODO(https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/2081):
	// Make this timeout configurable per-plugin or globally via the Director configuration to support plugins with
	// varying latency profiles.
	dataProducerTimeout       = 400 * time.Millisecond
	responseBodyQueueCapacity = 100
)

// Datastore defines the interface required by the Director.
type Datastore interface {
	PoolGet() (*datalayer.EndpointPool, error)
	ObjectiveGet(objectiveName string) *v1alpha2.InferenceObjective
	PodList(predicate func(fwkdl.Endpoint) bool) []fwkdl.Endpoint
	// ModelRewriteGet returns the rewrite rule for a given model name and the name of the InferenceModelRewrite object.
	ModelRewriteGet(modelName string) (*v1alpha2.InferenceModelRewriteRule, string)
}

// Scheduler defines the interface required by the Director for
type Scheduler interface {
	Schedule(ctx context.Context, request *fwkrhapi.InferenceRequest, candidateEndpoints []fwkrhapi.Endpoint) (result *fwkrhapi.SchedulingResult, err error)
}

// NewDirectorWithConfig creates a new Director instance with all dependencies.
func NewDirectorWithConfig(
	datastore Datastore,
	scheduler Scheduler,
	admissionController handlers.AdmissionController,
	endpointCandidates contracts.EndpointCandidates,
	config *Config,
) *Director {
	return &Director{
		datastore:             datastore,
		scheduler:             scheduler,
		admissionController:   admissionController,
		endpointCandidates:    endpointCandidates,
		requestControlPlugins: *config,
		defaultPriority:       0, // define default priority explicitly
	}
}

// responseBodyWork represents a unit of work to be processed by the async response body queue.
type responseBodyWork struct {
	ctx            context.Context
	request        *fwkrhapi.InferenceRequest
	response       *fwkrhapi.Response
	targetEndpoint *fwkdl.EndpointMetadata
}

// responseBodyQueue is a per-request async queue for processing response body plugin calls.
// It ensures chunks are processed in order via a channel while keeping plugin execution
// off the critical streaming path.
type responseBodyQueue struct {
	ch     chan responseBodyWork
	done   chan struct{} // closed when the processing goroutine exits
	mu     sync.Mutex
	closed bool
}

func newResponseBodyQueue() *responseBodyQueue {
	return &responseBodyQueue{
		ch:   make(chan responseBodyWork, responseBodyQueueCapacity),
		done: make(chan struct{}),
	}
}

func (q *responseBodyQueue) enqueue(work responseBodyWork) bool {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed {
		return false
	}
	q.ch <- work
	return true
}

func (q *responseBodyQueue) closeAndWait() {
	q.mu.Lock()
	if !q.closed {
		q.closed = true
		close(q.ch)
	}
	q.mu.Unlock()
	<-q.done
}

// Director orchestrates the request handling flow after initial parsing by the handler.
// Its responsibilities include:
// - Retrieving request metadata and relevant objectives.
// - Determining candidate pods.
// - Performing admission control via the AdmissionController.
// - Scheduling the request to target pod(s) via the Scheduler.
// - Running PreRequest plugins.
// - Preparing the request context for the Envoy ext_proc filter to route the request.
// - Running PostResponse plugins.
type Director struct {
	datastore             Datastore
	scheduler             Scheduler
	admissionController   handlers.AdmissionController
	endpointCandidates    contracts.EndpointCandidates
	requestControlPlugins Config
	// We just need a pointer to an int32 variable since Priority is a pointer in InferenceObjective.
	// No need to set this in the constructor, since the value we want is the default (0)
	defaultPriority    int
	responseBodyQueues sync.Map // map[*handlers.RequestContext]*responseBodyQueue
}

// HandleRequest is the main entry point for the Director to handle an incoming request.
func (d *Director) HandleRequest(ctx context.Context, reqCtx *handlers.RequestContext, inferenceRequestBody *fwkrhapi.InferenceRequestBody) (*handlers.RequestContext, error) {
	logger := log.FromContext(ctx)

	if err := d.modelRewriteIfNeeded(reqCtx, inferenceRequestBody); err != nil {
		return reqCtx, err
	}

	infObjective := d.datastore.ObjectiveGet(reqCtx.ObjectiveKey)
	priority := d.defaultPriority
	if infObjective != nil && infObjective.Spec.Priority != nil {
		priority = int(*infObjective.Spec.Priority)
	}

	reqCtx.Priority = priority

	requestObjectives := fwkrhapi.RequestObjectives{Priority: priority}

	// Prepare fwkrhapi.InferenceRequest (needed for both saturation detection and Scheduler)
	reqCtx.SchedulingRequest = &fwkrhapi.InferenceRequest{
		RequestID:        reqCtx.Request.Headers[reqcommon.RequestIDHeaderKey],
		TargetModel:      reqCtx.TargetModelName,
		Body:             inferenceRequestBody,
		Headers:          reqCtx.Request.Headers,
		Objectives:       requestObjectives,
		RequestSizeBytes: reqCtx.RequestSize,
	}

	logger = logger.WithValues("objectiveKey", reqCtx.ObjectiveKey, "incomingModelName", reqCtx.IncomingModelName, "targetModelName", reqCtx.TargetModelName, "priority", priority)
	ctx = log.IntoContext(ctx, logger)
	logger.V(logutil.DEBUG).Info("LLM request assembled")

	if err := d.admissionController.Admit(ctx, reqCtx, priority); err != nil {
		return reqCtx, err
	}

	endpointCandidates := d.endpointCandidates.Locate(ctx, reqCtx.Request.Metadata)
	if len(endpointCandidates) == 0 {
		return reqCtx, errcommon.Error{
			Code: errcommon.ServiceUnavailable,
			Msg:  "failed to find endpoint candidates for serving the request",
		}
	}

	snapshotOfCandidatePods := d.toSchedulerEndpoints(endpointCandidates)
	// Prepare per request data by running fwkrhapi.DataProducer plugins.
	if err := d.runDataProducerPlugins(ctx, reqCtx.SchedulingRequest, snapshotOfCandidatePods); err != nil {
		// Don't fail the request if fwkrhapi.DataProducer plugins fail.
		logger.Error(err, "failed to prepare per request data")
	}

	// Run admit request plugins
	if !d.runAdmissionPlugins(ctx, reqCtx.SchedulingRequest, snapshotOfCandidatePods) {
		return reqCtx, errcommon.Error{Code: errcommon.Internal, Msg: "request cannot be admitted"}
	}

	result, err := d.scheduler.Schedule(ctx, reqCtx.SchedulingRequest, snapshotOfCandidatePods)
	if err != nil {
		return reqCtx, errcommon.Error{Code: errcommon.ResourceExhausted, Msg: fmt.Errorf("failed to find target endpoint: %w", err).Error()}
	}

	reqCtx.SchedulingRequest.SchedulingResult = result

	// Prepare Request (Populates RequestContext and call fwkrhapi.PreRequest plugins)
	// Insert target endpoint to instruct Envoy to route requests to the specified target pod and attach the port number.
	// Invoke fwkrhapi.PreRequest registered plugins.
	reqCtx, err = d.prepareRequest(ctx, reqCtx, result)
	if err != nil {
		return reqCtx, err
	}
	if err := d.repackage(ctx, reqCtx, inferenceRequestBody); err != nil {
		return reqCtx, err
	}
	return reqCtx, nil
}

func (d *Director) modelRewriteIfNeeded(reqCtx *handlers.RequestContext, inferenceRequestBody *fwkrhapi.InferenceRequestBody) error {
	if v, ok := inferenceRequestBody.Payload.(fwkrhapi.PayloadMap); ok {
		// Mutate the model name inside the map, this is currently only supported if the payload is a fwkrhapi.PayloadMap.
		_, err := d.mutateModel(reqCtx, v)
		if err != nil {
			return err
		}
	}
	return nil
}

func (d *Director) mutateModel(reqCtx *handlers.RequestContext, bodyMap map[string]any) (*handlers.RequestContext, error) {
	var ok bool
	reqCtx.IncomingModelName, ok = bodyMap["model"].(string)
	if !ok {
		return reqCtx, errcommon.Error{Code: errcommon.BadRequest, Msg: "model not found in request body"}
	}
	if reqCtx.TargetModelName == "" {
		// Default to incoming model name
		reqCtx.TargetModelName = reqCtx.IncomingModelName
	}
	d.applyWeightedModelRewrite(reqCtx)
	bodyMap["model"] = reqCtx.TargetModelName
	return reqCtx, nil
}

func (d *Director) repackage(ctx context.Context, reqCtx *handlers.RequestContext, inferenceRequestBody *fwkrhapi.InferenceRequestBody) error {
	logger := log.FromContext(ctx)
	switch v := inferenceRequestBody.Payload.(type) {
	case fwkrhapi.PayloadMap:
		requestBodyBytes, err := json.Marshal(v)
		if err != nil {
			logger.Error(err, "Error marshalling request body")
			return errcommon.Error{Code: errcommon.Internal, Msg: "Error marshalling request body"}
		}
		reqCtx.Request.RawBody = requestBodyBytes
		reqCtx.RequestSize = len(requestBodyBytes)
	case fwkrhapi.PayloadProto, fwkrhapi.RawPayload:
		reqCtx.RequestSize = len(reqCtx.Request.RawBody)
	default:
		return errcommon.Error{Code: errcommon.BadRequest, Msg: "Unsupported llmRequest parsedBody"}
	}
	return nil
}

func (d *Director) applyWeightedModelRewrite(reqCtx *handlers.RequestContext) {
	rewriteRule, modelRewriteName := d.datastore.ModelRewriteGet(reqCtx.IncomingModelName)
	if rewriteRule == nil {
		return
	}
	reqCtx.TargetModelName = d.selectWeightedModel(rewriteRule.Targets)
	metrics.RecordInferenceModelRewriteDecision(modelRewriteName, reqCtx.IncomingModelName, reqCtx.TargetModelName)
}

func (d *Director) selectWeightedModel(models []v1alpha2.TargetModel) string {
	if len(models) == 0 {
		return ""
	}

	var totalWeight int32
	for _, model := range models {
		totalWeight += model.Weight
	}

	if totalWeight == 0 {
		// If total weight is 0, distribute evenly
		return models[rand.Intn(len(models))].ModelRewrite
	}

	randomNum := rand.Intn(int(totalWeight))
	var currentWeight int32
	for _, model := range models {
		currentWeight += model.Weight
		if randomNum < int(currentWeight) {
			return model.ModelRewrite
		}
	}

	// Should not happen
	return models[len(models)-1].ModelRewrite
}

// prepareRequest populates the RequestContext and calls the registered fwkrhapi.PreRequest plugins
// for allowing plugging customized logic based on the scheduling result.
func (d *Director) prepareRequest(ctx context.Context, reqCtx *handlers.RequestContext, result *fwkrhapi.SchedulingResult) (*handlers.RequestContext, error) {
	logger := log.FromContext(ctx)
	if result == nil || len(result.ProfileResults) == 0 {
		return reqCtx, errcommon.Error{Code: errcommon.Internal, Msg: "results must be greater than zero"}
	}
	// primary profile is used to set destination
	targetMetadatas := []*fwkdl.EndpointMetadata{}
	targetEndpoints := []string{}

	for _, pod := range result.ProfileResults[result.PrimaryProfileName].TargetEndpoints {
		curMetadata := pod.GetMetadata()
		curEndpoint := net.JoinHostPort(curMetadata.GetIPAddress(), curMetadata.GetPort())
		targetMetadatas = append(targetMetadatas, curMetadata)
		targetEndpoints = append(targetEndpoints, curEndpoint)
	}

	multiEndpointString := strings.Join(targetEndpoints, ",")
	logger.V(logutil.VERBOSE).Info("Request handled", "objectiveKey", reqCtx.ObjectiveKey, "incomingModelName", reqCtx.IncomingModelName, "targetModel", reqCtx.TargetModelName, "endpoint", multiEndpointString)

	reqCtx.TargetPod = targetMetadatas[0]
	reqCtx.TargetEndpoint = multiEndpointString

	d.runPreRequestPlugins(ctx, reqCtx.SchedulingRequest, result)

	return reqCtx, nil
}

func (d *Director) toSchedulerEndpoints(endpoints []fwkdl.Endpoint) []fwkrhapi.Endpoint {
	result := make([]fwkrhapi.Endpoint, len(endpoints))
	for i, endpoint := range endpoints {
		result[i] = fwkrhapi.NewEndpoint(endpoint.GetMetadata(), endpoint.GetMetrics(), endpoint.GetAttributes())
	}

	return result
}

// HandleResponseHeader is called when the response headers are received.
func (d *Director) HandleResponseHeader(ctx context.Context, reqCtx *handlers.RequestContext) *handlers.RequestContext {
	if len(d.requestControlPlugins.responseReceivedPlugins) == 0 {
		return reqCtx
	}
	response := &fwkrhapi.Response{
		RequestID:   reqCtx.Request.Headers[reqcommon.RequestIDHeaderKey],
		Headers:     reqCtx.Response.Headers,
		ReqMetadata: reqCtx.Request.Metadata,
	}
	// TODO: to extend fallback functionality, handle cases where target pod is unavailable
	// https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/1224
	d.runResponseHeaderPlugins(ctx, reqCtx.SchedulingRequest, response, reqCtx.TargetPod)
	return reqCtx
}

// HandleResponseBody is invoked by the director for every chunk received in a streaming
// response, or exactly once for a non-streaming response.
//
// For intermediate streaming chunks (endOfStream=false), the work is sent to a per-request
// async queue (channel + goroutine) so plugins run off the critical path while preserving
// chunk ordering. For the final chunk (endOfStream=true), the queue is drained first, then
// plugins run synchronously because they may produce DynamicMetadata that must be attached
// to the ext_proc response sent back to Envoy.
func (d *Director) HandleResponseBody(ctx context.Context, reqCtx *handlers.RequestContext, endOfStream bool) *handlers.RequestContext {
	logger := log.FromContext(ctx).WithValues("stage", "bodyChunk")
	logger.V(logutil.TRACE).Info("Entering HandleResponseBodyChunk")
	if len(d.requestControlPlugins.responseStreamingPlugins) == 0 {
		logger.V(logutil.TRACE).Info("Exiting HandleResponseBodyChunk")
		return reqCtx
	}

	startOfStream := !reqCtx.ResponseBodyStarted
	reqCtx.ResponseBodyStarted = true
	response := &fwkrhapi.Response{
		RequestID:     reqCtx.Request.Headers[reqcommon.RequestIDHeaderKey],
		Headers:       reqCtx.Response.Headers,
		StartOfStream: startOfStream,
		EndOfStream:   endOfStream,
		Usage:         reqCtx.Usage,
	}
	requestID := reqCtx.Request.Headers[reqcommon.RequestIDHeaderKey]

	if endOfStream {
		// Drain the async queue: close the channel and wait for the goroutine to finish
		// processing all previously queued chunks before running the final chunk synchronously.
		if val, ok := d.responseBodyQueues.LoadAndDelete(reqCtx); ok {
			q := val.(*responseBodyQueue)
			q.closeAndWait()
		}
		// Run the final chunk synchronously so DynamicMetadata is available for the response.
		d.runResponseBodyPlugins(ctx, reqCtx.SchedulingRequest, response, reqCtx.TargetPod)
		reqCtx.Response.DynamicMetadata = response.DynamicMetadata
	} else {
		// Get or create the async queue for this request.
		work := responseBodyWork{
			ctx:            ctx,
			request:        reqCtx.SchedulingRequest,
			response:       response,
			targetEndpoint: reqCtx.TargetPod,
		}
		q := d.loadOrCreateResponseBodyQueue(reqCtx)
		if !q.enqueue(work) {
			logger.V(logutil.DEBUG).Info("Skipping response body chunk because the async queue is closed", "requestID", requestID)
		}
	}
	logger.V(logutil.TRACE).Info("Exiting HandleResponseBodyChunk")
	return reqCtx
}

func (d *Director) GetRandomEndpoint() *fwkdl.EndpointMetadata {
	pods := d.datastore.PodList(nil)
	if len(pods) == 0 {
		return nil
	}
	return pods[rand.Intn(len(pods))].GetMetadata()
}

func (d *Director) loadOrCreateResponseBodyQueue(reqCtx *handlers.RequestContext) *responseBodyQueue {
	if val, ok := d.responseBodyQueues.Load(reqCtx); ok {
		return val.(*responseBodyQueue)
	}
	q := newResponseBodyQueue()
	if val, loaded := d.responseBodyQueues.LoadOrStore(reqCtx, q); loaded {
		return val.(*responseBodyQueue)
	}
	// Start the processing goroutine for this queue.
	go d.processResponseBodyQueue(reqCtx, q)
	return q
}

// processResponseBodyQueue reads work items from the queue channel and runs response body
// plugins asynchronously.
func (d *Director) processResponseBodyQueue(reqCtx *handlers.RequestContext, q *responseBodyQueue) {
	defer close(q.done)
	for work := range q.ch {
		d.runResponseBodyPlugins(work.ctx, work.request, work.response, work.targetEndpoint)
	}
}

func (d *Director) runDataProducerPlugins(ctx context.Context, request *fwkrhapi.InferenceRequest,
	endpoints []fwkrhapi.Endpoint) error {
	logger := log.FromContext(ctx)
	// Execute the registered fwkrhapi.DataProducer plugins concurrently.
	var wg sync.Map
	for _, plugin := range d.requestControlPlugins.dataProducerPlugins {
		wg.Store(plugin.TypedName().String(), struct{}{})
		go func(p fwkrhapi.DataProducer) {
			ctx, cancel := context.WithTimeout(ctx, dataProducerTimeout)
			defer cancel()
			defer wg.Delete(p.TypedName().String())
			before := time.Now()
			if err := p.Produce(ctx, request, endpoints); err != nil {
				logger.Error(err, "failed to produce data", "plugin", p.TypedName())
			}
			metrics.RecordPluginProcessingLatency("DataProducer", p.TypedName().Type, p.TypedName().Name, time.Since(before))
		}(plugin)
	}
	// TODO: wait for all plugins to finish or timeout.
	return nil
}

func (d *Director) runAdmissionPlugins(ctx context.Context, request *fwkrhapi.InferenceRequest, endpoints []fwkrhapi.Endpoint) bool {
	logger := log.FromContext(ctx)
	for _, plugin := range d.requestControlPlugins.admissionPlugins {
		before := time.Now()
		if err := plugin.AdmitRequest(ctx, request, endpoints); err != nil {
			logger.Error(err, "request rejected", "plugin", plugin.TypedName())
			return false
		}
		metrics.RecordPluginProcessingLatency("Admitter", plugin.TypedName().Type, plugin.TypedName().Name, time.Since(before))
	}
	return true
}

func (d *Director) runPreRequestPlugins(ctx context.Context, request *fwkrhapi.InferenceRequest,
	schedulingResult *fwkrhapi.SchedulingResult) {
	// PreRequest plugins are executed in parallel.
	for _, plugin := range d.requestControlPlugins.preRequestPlugins {
		go func(p fwkrhapi.PreRequest) {
			before := time.Now()
			p.PreRequest(ctx, request, schedulingResult)
			metrics.RecordPluginProcessingLatency("PreRequest", p.TypedName().Type, p.TypedName().Name, time.Since(before))
		}(plugin)
	}
}

func (d *Director) runResponseHeaderPlugins(ctx context.Context, request *fwkrhapi.InferenceRequest, response *fwkrhapi.Response, targetEndpoint *fwkdl.EndpointMetadata) {
	for _, plugin := range d.requestControlPlugins.responseReceivedPlugins {
		before := time.Now()
		plugin.ResponseHeader(ctx, request, response, targetEndpoint)
		metrics.RecordPluginProcessingLatency("ResponseReceived", plugin.TypedName().Type, plugin.TypedName().Name, time.Since(before))
	}
}

func (d *Director) runResponseBodyPlugins(ctx context.Context, request *fwkrhapi.InferenceRequest, response *fwkrhapi.Response, targetEndpoint *fwkdl.EndpointMetadata) {
	loggerTrace := log.FromContext(ctx).V(logutil.TRACE)
	for _, plugin := range d.requestControlPlugins.responseStreamingPlugins {
		loggerTrace.Info("Starting running ResponseStreaming plugin", "plugin", plugin.TypedName())
		before := time.Now()
		plugin.ResponseBody(ctx, request, response, targetEndpoint)
		metrics.RecordPluginProcessingLatency("ResponseStreaming", plugin.TypedName().Type, plugin.TypedName().Name, time.Since(before))
		loggerTrace.Info("Completed running ResponseStreaming plugin successfully", "plugin", plugin.TypedName())
	}
}
