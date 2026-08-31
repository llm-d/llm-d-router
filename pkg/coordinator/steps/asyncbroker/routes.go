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

package asyncbroker

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/go-chi/chi/v5"
	"github.com/llm-d/llm-d-async/api"
	"github.com/redis/go-redis/v9"
)

// validAsyncRequestID mirrors the server's x-request-id validation, so the
// fetch routes accept exactly the ids the inference path can produce.
var validAsyncRequestID = regexp.MustCompile(`^[a-zA-Z0-9\-]{1,128}$`)

// resultFetchGraceTTL is the default mailbox expiry applied after a
// successful fetch delivery, overridable via fetch_grace_seconds. The full
// result TTL exists for the completion-to-first-fetch window, which is
// client paced. After a delivered fetch, retention only covers lost-response
// retries, which fire within seconds, so the key is shrunk to this grace
// window instead of lingering for the full TTL.
const resultFetchGraceTTL = 60 * time.Second

// RegisterRoutes serves the async result lifecycle from the coordinator
// listener: fetch and delete for queued results, and the OpenAI models list
// derived from the configured routes. These handlers never run the pipeline;
// they are plain Redis reads and writes.
func (s *Step) RegisterRoutes(r chi.Router) {
	r.Get("/v1/models", s.handleModels)
	r.Get("/v1/requests/{id}", s.handleFetch)
	r.Delete("/v1/requests/{id}", s.handleDelete)
}

func (s *Step) tenantOf(r *http.Request) string {
	if t := r.Header.Get(s.cfg.TenantHeader); t != "" {
		return t
	}
	return defaultAsyncTenant
}

func (s *Step) handleModels(w http.ResponseWriter, _ *http.Request) {
	type model struct {
		ID      string `json:"id"`
		Object  string `json:"object"`
		OwnedBy string `json:"owned_by"`
	}
	seen := map[string]bool{}
	models := []model{}
	for _, rt := range s.cfg.Routes {
		if rt.Model == "" || seen[rt.Model] {
			continue
		}
		seen[rt.Model] = true
		models = append(models, model{ID: rt.Model, Object: "model", OwnedBy: "llm-d-async"})
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": models})
}

// checkFetchParams applies the enqueue path's tenant and id validation to
// the fetch/delete routes, so the read side never composes a Redis key the
// write side could not have produced. Writes the 400 itself and reports
// whether the request may proceed.
func (s *Step) checkFetchParams(w http.ResponseWriter, tenant, id string) bool {
	if strings.Contains(tenant, ":") {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", api.ErrCodeInvalidRequest,
			fmt.Sprintf("%s must not contain %q", s.cfg.TenantHeader, ":"))
		return false
	}
	if !validAsyncRequestID.MatchString(id) {
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", api.ErrCodeInvalidRequest,
			"request id must be 1-128 alphanumeric or dash characters")
		return false
	}
	return true
}

func (s *Step) handleFetch(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	tenant := s.tenantOf(r)
	if !s.checkFetchParams(w, tenant, id) {
		return
	}
	state, res, err := lookupResult(r.Context(), s.rdb, tenant, id)
	if err != nil {
		s.logger.Error(err, "result lookup failed", "id", id)
		writeOpenAIError(w, http.StatusInternalServerError, "api_error", "LOOKUP_FAILED",
			"failed to look up the request")
		return
	}
	switch state {
	case asyncStateReady:
		// Fetch stays non-destructive, but a confirmed delivery shrinks the
		// mailbox TTL to the retry grace window. Deleting outright would race
		// the client's retry of a response lost past the coordinator's write.
		// DELETE remains the immediate reclaim for tidy clients.
		if writeResult(w, res) == nil {
			graceCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if grace := s.cfg.fetchGrace(); grace > 0 {
				if err := s.rdb.Expire(graceCtx, resultKey(tenant, id), grace).Err(); err != nil {
					s.logger.Error(err, "failed to shrink result ttl after fetch", "id", id)
				}
			} else if err := s.rdb.Del(graceCtx, resultKey(tenant, id)).Err(); err != nil {
				s.logger.Error(err, "failed to delete result after fetch", "id", id)
			}
		}
	case asyncStatePending:
		writePending(w, id)
	default:
		writeOpenAIError(w, http.StatusGone, "invalid_request_error", "UNKNOWN_REQUEST",
			"request id is unknown, expired, or already deleted")
	}
}

func (s *Step) handleDelete(w http.ResponseWriter, r *http.Request) {
	id := chi.URLParam(r, "id")
	tenant := s.tenantOf(r)
	if !s.checkFetchParams(w, tenant, id) {
		return
	}
	// Cancel first so a still-queued request is dropped pre-dispatch instead
	// of running to completion and recreating the mailbox this handler just
	// deleted. Best effort: a request already dispatched runs to completion,
	// and its result then sits out the mailbox TTL.
	if err := s.sub.CancelRequests(r.Context(), []string{envelopeID(tenant, id)}); err != nil {
		s.logger.Error(err, "failed to cancel request on delete", "id", id)
	}
	if err := s.rdb.Del(r.Context(), resultKey(tenant, id)).Err(); err != nil {
		s.logger.Error(err, "failed to delete result", "id", id)
		writeOpenAIError(w, http.StatusInternalServerError, "api_error", "DELETE_FAILED",
			"failed to delete the result")
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

// resultKey is the per-request result destination, scoped by tenant so ids
// cannot collide across tenants. Tenant header is trusted as asserted (see the
// deployment notes in docs/coordinator_async_broker.md). Messages enqueued by the
// step set result_queue_name to this key, and the AP's result writer
// delivers there. Fetch reads are non-destructive: a delivered fetch shrinks
// the TTL to resultFetchGraceTTL (covering lost-response retries), DELETE
// reclaims immediately, and the full TTL bounds the unfetched case. Wait
// mode deletes the key once the result is delivered on the held connection,
// its only consumer.
func resultKey(tenant, id string) string {
	return resultKeyPrefix + tenant + ":" + id
}

// envelopeID is the request id written into the AP envelope: the client id
// prefixed with the tenant. The AP derives its per-request keys (active
// marker, cancel key) from the envelope id, so prefixing here is what tenant
// scopes them. The bare client id stays the only externally visible form.
func envelopeID(tenant, id string) string {
	return tenant + ":" + id
}

// asyncRequestState is the outcome of looking up a request id.
type asyncRequestState int

const (
	asyncStateReady asyncRequestState = iota
	asyncStatePending
	asyncStateUnknown
)

// lookupResult reads a request's result without consuming it. Pending is
// detected via the producer-maintained active-token key, which exists from
// submit until the result is flushed. Both reads are tenant-scoped, so a
// caller with the wrong tenant sees an unknown id.
func lookupResult(ctx context.Context, rdb *redis.Client, tenant, id string) (asyncRequestState, *api.ResultMessage, error) {
	res, found, err := readMailbox(ctx, rdb, tenant, id)
	if err != nil {
		return asyncStateUnknown, nil, err
	}
	if found {
		return asyncStateReady, res, nil
	}

	exists, err := rdb.Exists(ctx, api.RequestActiveTokenKey(envelopeID(tenant, id))).Result()
	if err != nil {
		return asyncStateUnknown, nil, fmt.Errorf("failed to check request state: %w", err)
	}
	if exists > 0 {
		return asyncStatePending, nil, nil
	}
	// The AP can deliver the result and clear the active marker between the
	// two reads above, which would misread a just-completed request as
	// unknown: re-read the mailbox once before concluding that.
	res, found, err = readMailbox(ctx, rdb, tenant, id)
	if err != nil {
		return asyncStateUnknown, nil, err
	}
	if found {
		return asyncStateReady, res, nil
	}
	return asyncStateUnknown, nil, nil
}

// readMailbox returns the stored result, reporting whether one was present.
func readMailbox(ctx context.Context, rdb *redis.Client, tenant, id string) (*api.ResultMessage, bool, error) {
	vals, err := rdb.LRange(ctx, resultKey(tenant, id), 0, 0).Result()
	if err != nil && !errors.Is(err, redis.Nil) {
		return nil, false, fmt.Errorf("failed to read result: %w", err)
	}
	if len(vals) == 0 {
		return nil, false, nil
	}
	var res api.ResultMessage
	if err := json.Unmarshal([]byte(vals[0]), &res); err != nil {
		return nil, false, fmt.Errorf("failed to parse stored result: %w", err)
	}
	return &res, true, nil
}

// openAIError is the OpenAI error response envelope.
type openAIError struct {
	Error openAIErrorBody `json:"error"`
}

type openAIErrorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
	Code    string `json:"code,omitempty"`
}

func writeOpenAIError(w http.ResponseWriter, status int, errType, code, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(openAIError{Error: openAIErrorBody{Message: msg, Type: errType, Code: code}})
}

func writePending(w http.ResponseWriter, id string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusAccepted)
	_ = json.NewEncoder(w).Encode(map[string]string{"id": id, "status": "pending"})
}

// writeResult maps a stored ResultMessage onto the HTTP response. StatusCode
// > 0 mirrors the upstream status and body verbatim (out-of-range codes
// answer 502 instead). Error codes map:
// GATE_DROPPED -> 429, DEADLINE_EXCEEDED -> 504, INVALID_REQUEST -> 400,
// CANCELLED -> 499, everything else -> 502, wrapped in the OpenAI error
// envelope. Returns the body write error so callers that confirm delivery
// can act on it; the error-envelope branches always return nil.
func writeResult(w http.ResponseWriter, res *api.ResultMessage) error {
	if res.StatusCode > 0 {
		// The stored result is external data: guard the range rather than
		// letting WriteHeader panic on a corrupted value.
		if res.StatusCode < 100 || res.StatusCode > 599 {
			writeOpenAIError(w, http.StatusBadGateway, "api_error", "MALFORMED_RESULT",
				fmt.Sprintf("stored result carries invalid status code %d", res.StatusCode))
			return nil
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(res.StatusCode)
		_, err := w.Write([]byte(res.Payload))
		return err
	}
	switch res.ErrorCode {
	case api.ErrCodeGateDropped:
		writeOpenAIError(w, http.StatusTooManyRequests, "rate_limit_error", res.ErrorCode, res.ErrorMessage)
	case api.ErrCodeDeadlineExceeded:
		writeOpenAIError(w, http.StatusGatewayTimeout, "timeout_error", res.ErrorCode, res.ErrorMessage)
	case api.ErrCodeInvalidRequest:
		writeOpenAIError(w, http.StatusBadRequest, "invalid_request_error", res.ErrorCode, res.ErrorMessage)
	case api.ErrCodeCancelled:
		// 499 Client Closed Request (nginx convention): the caller abandoned
		// the request and cancellation dropped it pre-dispatch.
		writeOpenAIError(w, 499, "cancelled", res.ErrorCode, res.ErrorMessage)
	default:
		writeOpenAIError(w, http.StatusBadGateway, "api_error", res.ErrorCode, res.ErrorMessage)
	}
	return nil
}
