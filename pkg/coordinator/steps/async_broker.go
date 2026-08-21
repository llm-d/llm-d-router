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

package steps

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"github.com/llm-d/llm-d-async/api"
	"github.com/llm-d/llm-d-async/producer"
	"github.com/redis/go-redis/v9"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const AsyncBrokerStepName = "async-broker"

func init() {
	pipeline.Register(AsyncBrokerStepName, NewAsyncBrokerStep)
}

// asyncSubmitter is the subset of producer.Producer the step needs.
type asyncSubmitter interface {
	SubmitRequest(ctx context.Context, req api.Request) error
	CancelRequests(ctx context.Context, requestIDs []string) error
}

// waitPollInterval is the poll cadence when the multiplexed wake-up is
// unavailable (Redis without keyspace notifications, or wakeup_mode: poll).
const waitPollInterval = 200 * time.Millisecond

// waitBackupPollInterval is the slow safety poll under the notify wake-up:
// keyspace notifications are fire and forget, so a lost notification is
// recovered on the next backup tick rather than never.
const waitBackupPollInterval = 2 * time.Second

// AsyncBrokerStep bridges the coordinator to the llm-d-async broker. A
// request that carries the mode header opts into async serving: passthrough
// labels it (quota classification, objective and fairness headers) and lets
// the pipeline continue; enqueue and wait hand it to the async processor via
// the broker queue and serve the response themselves. A request without the
// mode header is left completely untouched, which is also what keeps
// AP-dispatched requests inert on re-entry. The step must run before the
// pipeline's processing steps, and it registers the result retrieval routes
// (GET/DELETE /v1/requests/{id}, GET /v1/models) on the coordinator listener.
type AsyncBrokerStep struct {
	cfg    *asyncBrokerConfig
	rdb    *redis.Client
	sub    asyncSubmitter
	quota  *asyncQuotaClassifier
	waiter *asyncResultWaiter // nil = poll mode
	// logger serves construction-time messages and the registered route
	// handlers, which run outside a pipeline request context.
	logger logr.Logger

	// unroutedModels dedupes the unrouted-model log line per model name so
	// the first occurrence stays visible without flooding under load. Model
	// names arrive in request bodies, so the set is size-capped; see
	// rememberUnrouted.
	unroutedMu     sync.Mutex
	unroutedModels map[string]struct{}
}

var _ pipeline.Step = (*AsyncBrokerStep)(nil)

// NewAsyncBrokerStep builds the step from its params block. It connects to
// the Redis broker at construction; wakeup_mode auto probes the server's
// keyspace-notification support once at startup.
func NewAsyncBrokerStep(_ *gateway.Client, params map[string]any) (pipeline.Step, error) {
	cfg, err := parseAsyncBrokerConfig(params)
	if err != nil {
		return nil, fmt.Errorf("async-broker: %w", err)
	}
	opts, err := redis.ParseURL(cfg.RedisURL)
	if err != nil {
		return nil, fmt.Errorf("async-broker: invalid redis_url: %w", err)
	}
	rdb := redis.NewClient(opts)

	prod, err := producer.NewRedisSortedSetProducer(producer.RedisSortedSetConfig{
		RequestQueueName: cfg.DefaultQueue,
		// Placeholder: every message sets its own per-request result key.
		ResultQueueName: resultKeyPrefix + "unrouted",
	}, producer.WithRedisClient(rdb))
	if err != nil {
		return nil, fmt.Errorf("async-broker: failed to create producer: %w", err)
	}

	logger := ctrl.Log.WithName(AsyncBrokerStepName)
	s := &AsyncBrokerStep{
		cfg:            cfg,
		rdb:            rdb,
		sub:            prod,
		quota:          &asyncQuotaClassifier{rdb: rdb, cfg: cfg.Quota},
		logger:         logger,
		unroutedModels: map[string]struct{}{},
	}
	switch cfg.WakeupMode {
	case "poll":
	case "notify":
		s.waiter = newAsyncResultWaiter(context.Background(), rdb, opts.DB, logger)
	default: // auto
		detectCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		enabled := asyncNotificationsEnabled(detectCtx, rdb)
		cancel()
		if enabled {
			s.waiter = newAsyncResultWaiter(context.Background(), rdb, opts.DB, logger)
			logger.Info("wait mode using keyspace-notification wake-up")
		} else {
			logger.Info("Redis keyspace notifications not enabled (notify-keyspace-events needs K and l), wait mode falls back to polling")
		}
	}
	return s, nil
}

func (s *AsyncBrokerStep) Name() string { return AsyncBrokerStepName }

// maxTrackedUnroutedModels bounds the unrouted-model dedup set. Model names
// are client-controlled, so an unbounded set would let unique junk names
// grow memory without limit; past the cap, repeats simply log at debug.
const maxTrackedUnroutedModels = 1024

// routeFor resolves the request's queue and tier, surfacing fall-through to
// the defaults. Serving is unchanged — the defaults exist for exactly this
// case — but an unrouted model must be visible somewhere: a typo in a route,
// or a model served under a name no route lists (an adapter id whose base
// model is the one routed), otherwise sends every request to the default
// queue and tier with each response still 200, indistinguishable from a
// correctly routed deployment. The first unrouted request per model logs at
// Info; repeats log at debug.
func (s *AsyncBrokerStep) routeFor(ctx context.Context, model, tenant string) (queue, tier string) {
	queue, tier, matched := s.cfg.route(model, tenant)
	if matched {
		return queue, tier
	}
	logger := log.FromContext(ctx).WithName(AsyncBrokerStepName)
	if s.rememberUnrouted(model) {
		logger.Info("no route matched, serving from defaults",
			"model", model, "tenant", tenant, "queue", queue, "tier", tier)
	} else {
		logger.V(logutil.DEBUG).Info("no route matched, serving from defaults",
			"model", model, "tenant", tenant, "queue", queue, "tier", tier)
	}
	return queue, tier
}

// rememberUnrouted records the model in the dedup set, reporting true when
// this is its first occurrence and the set has room.
func (s *AsyncBrokerStep) rememberUnrouted(model string) bool {
	s.unroutedMu.Lock()
	defer s.unroutedMu.Unlock()
	if _, seen := s.unroutedModels[model]; seen {
		return false
	}
	if len(s.unroutedModels) >= maxTrackedUnroutedModels {
		return false
	}
	s.unroutedModels[model] = struct{}{}
	return true
}

func (s *AsyncBrokerStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	headerVal := reqCtx.OriginalHeaders.Get(s.cfg.ModeHeader)
	if headerVal == "" {
		// No mode header means the request did not opt into async serving:
		// leave it untouched. AP dispatches re-enter the pipeline this way,
		// since the step never forwards the mode header onto queued messages.
		return nil
	}

	w := reqCtx.ResponseWriter
	mode := asyncMode(strings.ToLower(headerVal))
	switch mode {
	case asyncModePassthrough, asyncModeEnqueue, asyncModeWait:
	default:
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", api.ErrCodeInvalidRequest,
			fmt.Sprintf("unknown %s value", s.cfg.ModeHeader))
		return pipeline.ErrPipelineDone
	}
	if reqCtx.Stream && mode != asyncModePassthrough {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", api.ErrCodeInvalidRequest,
			"stream is only supported in passthrough mode")
		return pipeline.ErrPipelineDone
	}
	if reqCtx.Model == "" {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", api.ErrCodeInvalidRequest,
			"model is required")
		return pipeline.ErrPipelineDone
	}
	// The result key joins tenant and id with ":", so a ":" in the tenant
	// would alias another (tenant, id) pair's key and leak results across
	// tenants. The request id is server-validated to alphanumerics and
	// dashes, so it cannot collide.
	tenant := reqCtx.OriginalHeaders.Get(s.cfg.TenantHeader)
	if tenant == "" {
		tenant = defaultAsyncTenant
	}
	if strings.Contains(tenant, ":") {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", api.ErrCodeInvalidRequest,
			fmt.Sprintf("%s must not contain %q", s.cfg.TenantHeader, ":"))
		return pipeline.ErrPipelineDone
	}

	if mode == asyncModePassthrough {
		return s.passthrough(ctx, reqCtx, tenant)
	}
	w.Header().Set(reqcommon.RequestIDHeaderKey, reqCtx.RequestID)
	return s.serveQueued(ctx, reqCtx, mode, tenant)
}

// passthrough labels the request and lets the pipeline serve it: quota
// classification against the shared Redis counters selects the reserved or
// overflow objective for the request's tier, and the fairness header carries
// the tenant. The stamped headers reach the EPP on every phase call via
// ForwardedHeaders.
func (s *AsyncBrokerStep) passthrough(ctx context.Context, reqCtx *pipeline.RequestContext, tenant string) error {
	classification, release, err := s.quota.classify(ctx, tenant)
	if err != nil {
		log.FromContext(ctx).WithName(AsyncBrokerStepName).Error(err, "quota classification failed open", "tenant", tenant)
	}
	if release != nil {
		// Hold the reserved slot until the request finishes: the request
		// context is canceled when the handler returns or the client
		// disconnects, mirroring the redis-quota gate's release-on-completion
		// semantics.
		go func() {
			<-ctx.Done()
			release()
		}()
	}

	_, tier := s.routeFor(ctx, reqCtx.Model, tenant)
	// Identity headers are step-owned: strip client-supplied values
	// unconditionally so priority cannot be self-assigned, even for tiers
	// without a configured objective mapping.
	reqCtx.OriginalHeaders.Del(s.cfg.ObjectiveHeader)
	reqCtx.OriginalHeaders.Del(s.cfg.FairnessHeader)
	if pair, ok := s.cfg.Objectives[tier]; ok {
		objective := pair.Reserved
		if classification == asyncClassificationOverflow {
			objective = pair.Overflow
		}
		if objective != "" {
			reqCtx.OriginalHeaders.Set(s.cfg.ObjectiveHeader, objective)
		}
	}
	reqCtx.OriginalHeaders.Set(s.cfg.FairnessHeader, tenant)
	return nil
}

// serveQueued enqueues onto the broker with a per-request result key. Mode
// enqueue responds 202 immediately. Mode wait holds the connection up to the
// wait cap, then falls back to the 202 response.
func (s *AsyncBrokerStep) serveQueued(ctx context.Context, reqCtx *pipeline.RequestContext, mode asyncMode, tenant string) error {
	w := reqCtx.ResponseWriter

	bounds := s.cfg.timeoutBounds(mode)
	timeoutSecs := bounds.DefaultSeconds
	if v := reqCtx.OriginalHeaders.Get(s.cfg.TimeoutHeader); v != "" {
		if parsed, err := strconv.ParseInt(v, 10, 64); err == nil && parsed > 0 {
			timeoutSecs = parsed
		}
	}
	if bounds.MaxSeconds > 0 && timeoutSecs > bounds.MaxSeconds {
		timeoutSecs = bounds.MaxSeconds
	}
	timeout := time.Duration(timeoutSecs) * time.Second

	queue, _ := s.routeFor(ctx, reqCtx.Model, tenant)
	now := time.Now()

	metadata := map[string]string{s.cfg.Quota.Attribute: tenant}
	if tp := reqCtx.OriginalHeaders.Get("traceparent"); tp != "" {
		metadata["traceparent"] = tp
	}
	var fwd map[string]string
	for _, h := range s.cfg.ForwardHeaders {
		if v := reqCtx.OriginalHeaders.Get(h); v != "" {
			if fwd == nil {
				fwd = map[string]string{}
			}
			fwd[h] = v
		}
	}

	msg := &api.RedisRequest{
		RequestMessage: api.RequestMessage{
			ID:       envelopeID(tenant, reqCtx.RequestID),
			Created:  now.Unix(),
			Deadline: now.Add(timeout).Unix(),
			Payload:  reqCtx.Body,
			Metadata: metadata,
			Headers:  fwd,
			Endpoint: reqCtx.OriginalPath,
		},
		RequestQueueName: queue,
		ResultQueueName:  resultKey(tenant, reqCtx.RequestID),
	}
	if err := s.sub.SubmitRequest(ctx, msg); err != nil {
		writeOpenAIError(w, http.StatusServiceUnavailable, "api_error", "ENQUEUE_FAILED",
			fmt.Sprintf("failed to enqueue request: %v", err))
		return pipeline.ErrPipelineDone
	}

	if mode == asyncModeEnqueue {
		writePending(w, reqCtx.RequestID)
		return pipeline.ErrPipelineDone
	}
	s.waitForResult(ctx, reqCtx, tenant, timeout)
	return pipeline.ErrPipelineDone
}

func (s *AsyncBrokerStep) waitForResult(ctx context.Context, reqCtx *pipeline.RequestContext, tenant string, timeout time.Duration) {
	logger := log.FromContext(ctx).WithName(AsyncBrokerStepName)
	w := reqCtx.ResponseWriter
	id := reqCtx.RequestID

	// The hold runs to the request deadline, or the configured wait cap if
	// that is shorter. holdIsDeadline selects the ending: 504 at the
	// deadline, 202 fallback at the cap.
	waitCap := timeout
	holdIsDeadline := true
	if s.cfg.WaitCapSeconds > 0 {
		if capDur := time.Duration(s.cfg.WaitCapSeconds) * time.Second; capDur < waitCap {
			waitCap = capDur
			holdIsDeadline = false
		}
	}
	// Held connections must outlive the server WriteTimeout, so clear the
	// write deadline as the streaming path does.
	if err := http.NewResponseController(w).SetWriteDeadline(time.Time{}); err != nil && !errors.Is(err, http.ErrNotSupported) {
		logger.V(logutil.DEFAULT).Info("could not clear write deadline for held connection", "error", err)
	}
	waitCtx, cancel := context.WithTimeout(ctx, waitCap)
	defer cancel()

	// Multiplexed wake-up: subscribe before checking so a result landing
	// between check and park still notifies. Falls back to polling when the
	// waiter is unavailable, or per registration error.
	var wake <-chan struct{}
	pollEvery := waitPollInterval
	if s.waiter != nil {
		if ch, cleanup, err := s.waiter.register(waitCtx, resultKey(tenant, id)); err == nil {
			defer cleanup()
			wake = ch
			pollEvery = waitBackupPollInterval
		} else {
			logger.V(logutil.DEFAULT).Info("wake-up registration failed, polling instead", "error", err)
		}
	}

	ticker := time.NewTicker(pollEvery)
	defer ticker.Stop()
	for {
		state, res, err := lookupResult(waitCtx, s.rdb, tenant, id)
		if err == nil && state == asyncStateReady {
			if writeResult(w, res) == nil {
				// Delivery confirmed on the held connection, the result's only
				// consumer: reclaim the mailbox now instead of letting it sit
				// out the full result TTL. On write failure the key stays, as
				// the client may still fetch by id.
				delCtx, delCancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer delCancel()
				if err := s.rdb.Del(delCtx, resultKey(tenant, id)).Err(); err != nil {
					logger.V(logutil.DEFAULT).Info("failed to delete delivered result", "id", id, "error", err)
				}
			}
			return
		}
		select {
		case <-waitCtx.Done():
			if ctx.Err() != nil {
				// Client disconnected: nobody will fetch this result, so
				// cancel pre-dispatch (best effort, per the producer's
				// cancellation contract). A client retry with the same id
				// re-submits cleanly: SubmitRequest clears stale markers.
				cancelCtx, cancelFn := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancelFn()
				if err := s.sub.CancelRequests(cancelCtx, []string{envelopeID(tenant, id)}); err != nil {
					logger.V(logutil.DEFAULT).Info("failed to cancel abandoned request", "id", id, "error", err)
				}
				return
			}
			if holdIsDeadline {
				// Past the deadline the AP refuses the work at gate, pop,
				// and send, so no result can arrive: answer 504 after a
				// final mailbox check for a boundary arrival.
				finalCtx, finalCancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer finalCancel()
				if state, res, err := lookupResult(finalCtx, s.rdb, tenant, id); err == nil && state == asyncStateReady {
					if writeResult(w, res) == nil {
						if err := s.rdb.Del(finalCtx, resultKey(tenant, id)).Err(); err != nil {
							logger.V(logutil.DEFAULT).Info("failed to delete delivered result", "id", id, "error", err)
						}
					}
					return
				}
				writeOpenAIError(w, http.StatusGatewayTimeout, "timeout_error", api.ErrCodeDeadlineExceeded,
					"request deadline exceeded before a result was produced")
				return
			}
			// Cap reached: fall back to the enqueue response, the request
			// stays queued and fetchable.
			writePending(w, id)
			return
		case <-wake:
		case <-ticker.C:
		}
	}
}
