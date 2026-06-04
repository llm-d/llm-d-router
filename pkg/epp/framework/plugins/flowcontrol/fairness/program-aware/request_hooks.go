package programaware

import (
	"context"
	"time"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// --- DataProducer interface ---

// Produces declares what data this plugin produces. No endpoint attributes are produced.
func (p *ProgramAwarePlugin) Produces() map[plugin.DataKey]any {
	return map[plugin.DataKey]any{}
}

// Consumes declares what data this plugin requires from other plugins. None.
func (p *ProgramAwarePlugin) Consumes() plugin.DataDependencies {
	return plugin.DataDependencies{}
}

// Produce reads the program ID from the request and increments the program's
// total request count. The enqueue timestamp for wait time calculation is
// recorded by Pick() (in flow control), not here, since Produce runs after
// the request has already left the flow control queue.
func (p *ProgramAwarePlugin) Produce(ctx context.Context, request *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	programID := request.FairnessID
	if programID == "" {
		programID = metadata.DefaultFairnessID
	}

	metrics := p.getOrCreateMetrics(programID)
	metrics.IncrementRequests()
	requestsTotal.WithLabelValues(programID).Inc()

	log.FromContext(ctx).V(logutil.TRACE).Info("Produce: recorded program request",
		"requestId", request.RequestID, "programId", programID,
		"totalRequests", metrics.TotalRequests())

	return nil
}

// --- PreRequest interface ---

// PreRequest is called after scheduling and before the request is sent to the model server.
// It calculates the wait time (enqueue → now) and updates the program's EWMA.
// The enqueue timestamp was recorded by Pick() during flow control dispatch.
func (p *ProgramAwarePlugin) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, _ *fwksched.SchedulingResult) {
	programID := request.FairnessID
	if programID == "" {
		programID = metadata.DefaultFairnessID
	}

	metrics := p.getOrCreateMetrics(programID)
	p.getStrategy().OnPreRequest(metrics, request)

	metrics.IncrementDispatched()
	metrics.IncrementInFlight()
	dispatchedTotal.WithLabelValues(programID).Inc()

	if enqueueTime, ok := fwksched.ReadRequestAttribute[time.Time](request, enqueueTimeAttributeKey); ok {
		waitMs := float64(time.Since(enqueueTime).Milliseconds())
		metrics.RecordWaitTime(waitMs)
		ewmaWaitTimeMs.WithLabelValues(programID).Set(metrics.AverageWaitTime())

		log.FromContext(ctx).V(logutil.TRACE).Info("PreRequest: recorded wait time",
			"requestId", request.RequestID, "programId", programID,
			"waitMs", waitMs, "avgWaitMs", metrics.AverageWaitTime())
	}
}

// --- ResponseComplete interface ---

// ResponseBody records token usage on the final stream chunk. For streaming
// responses ResponseBody fires once per chunk; only the final invocation
// (response.EndOfStream == true) carries the terminal Usage and is treated
// as the request-lifecycle hook.
func (p *ProgramAwarePlugin) ResponseBody(ctx context.Context, request *fwksched.InferenceRequest, response *fwkrc.Response, _ *datalayer.EndpointMetadata) {
	if request == nil || response == nil || !response.EndOfStream {
		return
	}
	programID := request.FairnessID
	if programID == "" {
		programID = metadata.DefaultFairnessID
	}

	metrics := p.getOrCreateMetrics(programID)
	metrics.DecrementInFlight()

	promptTokens := int64(response.Usage.PromptTokens)
	completionTokens := int64(response.Usage.CompletionTokens)

	metrics.RecordTokens(promptTokens, completionTokens)
	inputTokensTotal.WithLabelValues(programID).Add(float64(promptTokens))
	outputTokensTotal.WithLabelValues(programID).Add(float64(completionTokens))

	p.getStrategy().OnCompleted(metrics, request, response)
	attainedServiceTokens.WithLabelValues(programID).Set(metrics.AttainedService())

	// Update service rate for fairness index (weighted tokens/sec EWMA).
	cost := float64(weightInputToken*promptTokens + weightOutputToken*completionTokens)
	metrics.RecordServiceRate(cost, time.Now())
	serviceRateTokensPerSec.WithLabelValues(programID).Set(metrics.ServiceRate())

	log.FromContext(ctx).V(logutil.TRACE).Info("ResponseComplete: recorded tokens",
		"requestId", request.RequestID, "programId", programID,
		"promptTokens", promptTokens, "completionTokens", completionTokens,
		"attainedService", metrics.AttainedService())
}
