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
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

// asyncMode selects how an opted-in request is served.
type asyncMode string

const (
	// asyncModePassthrough labels the request (objective, fairness id,
	// quota classification) and lets the pipeline serve it normally.
	asyncModePassthrough asyncMode = "passthrough"
	// asyncModeEnqueue enqueues onto the broker and answers 202 with the
	// request id.
	asyncModeEnqueue asyncMode = "enqueue"
	// asyncModeWait enqueues and holds the connection until the result lands
	// or the wait cap expires (then falls back to the enqueue response).
	asyncModeWait asyncMode = "wait"
)

const (
	// defaultModeHeader selects the serving mode when mode_header is not
	// configured.
	defaultModeHeader = "X-AP-Mode"

	defaultAsyncTenant = "default"
	// resultKeyPrefix scopes the per-request result keys the async processor
	// delivers to; see resultKey.
	resultKeyPrefix = "results:req:"

	defaultAsyncTimeoutSecs   = 60
	defaultEnqueueTimeoutSecs = 3600
	// defaultWaitMaxSecs bounds client-requested wait deadlines when no
	// max_seconds is configured: every wait holds a live connection, so an
	// arbitrarily large requested deadline must not pin one for days.
	defaultWaitMaxSecs = 3600
)

// asyncRoute maps a request to a broker queue and tier. Empty Model or Tenant
// matches anything; the first matching route wins.
type asyncRoute struct {
	Model  string `json:"model,omitempty"`
	Tenant string `json:"tenant,omitempty"`
	Queue  string `json:"queue,omitempty"`
	Tier   string `json:"tier,omitempty"`
}

// asyncTimeoutBounds holds the default and maximum request deadline for a
// queued mode, in seconds.
type asyncTimeoutBounds struct {
	DefaultSeconds int64 `json:"default_seconds,omitempty"`
	MaxSeconds     int64 `json:"max_seconds,omitempty"`
}

// asyncObjectivePair holds the InferenceObjective names stamped for a tier,
// selected by the request's quota classification.
type asyncObjectivePair struct {
	Reserved string `json:"reserved,omitempty"`
	Overflow string `json:"overflow,omitempty"`
}

// asyncQuotaConfig configures passthrough quota classification. It mirrors
// the async processor's redis-quota gate concurrency mode and key scheme so
// the step and the AP's queue gates draw from the same counters. Queued modes
// are classified by the AP's gates at dequeue and are not counted here.
type asyncQuotaConfig struct {
	// Prefix + Attribute + ":" + tenant forms the counter key, matching the
	// redis-quota gate's fmt.Sprintf("%s%s:%s", prefix, attribute, value).
	Prefix    string `json:"prefix,omitempty"`
	Attribute string `json:"attribute,omitempty"`
	// Limits maps tenant name to reserved concurrency. Tenants without an
	// entry are always classified reserved.
	Limits map[string]int `json:"limits,omitempty"`
	// WindowSeconds is the counter TTL, matching the gate's window.
	WindowSeconds int `json:"window_seconds,omitempty"`
}

// asyncBrokerConfig is the async-broker step configuration, decoded from the
// step's params block.
type asyncBrokerConfig struct {
	RedisURL string `json:"redis_url"`

	// ModeHeader selects the serving mode per request (default X-AP-Mode).
	ModeHeader string `json:"mode_header,omitempty"`
	// TenantHeader names the header carrying the tenant key (default X-Team).
	TenantHeader string `json:"tenant_header,omitempty"`
	// TimeoutHeader lets clients request a deadline in seconds for the queued
	// modes (default X-Request-Timeout-Seconds). Requested deadlines are
	// clamped by the mode's max_seconds.
	TimeoutHeader string `json:"timeout_header,omitempty"`

	// Timeouts holds the deadline bounds per queued mode ("enqueue", "wait").
	// Each field falls back independently when unset: default_seconds to 60
	// for wait and 3600 for enqueue (deferred work legitimately outlives any
	// connection). An unset max_seconds means no clamp for enqueue, but wait
	// holds a live connection per request, so its max defaults to 3600 (or
	// default_seconds when that is configured higher).
	Timeouts map[string]asyncTimeoutBounds `json:"timeouts,omitempty"`
	// WaitCapSeconds, when > 0, bounds how long wait mode holds a connection
	// before falling back to the enqueue response. Unset, the hold runs to
	// the request deadline.
	WaitCapSeconds int64 `json:"wait_cap_seconds,omitempty"`
	// FetchGraceSeconds is the mailbox TTL applied after a delivered fetch,
	// covering lost-response retries. Unset falls back to
	// resultFetchGraceTTL. Zero deletes the result on delivery, matching
	// wait mode's eager delete.
	FetchGraceSeconds *int64 `json:"fetch_grace_seconds,omitempty"`

	// Routes select the broker queue and tier per request. The queues must
	// not set result_queue_name in the AP's queue config, so the per-request
	// result key routing applies.
	Routes       []asyncRoute `json:"routes,omitempty"`
	DefaultQueue string       `json:"default_queue,omitempty"`
	DefaultTier  string       `json:"default_tier,omitempty"`

	Quota asyncQuotaConfig `json:"quota,omitempty"`

	// ForwardHeaders lists client headers copied onto queued messages and
	// forwarded at dispatch. Defaults to the gateway SLO ordering headers.
	// The mode header and identity headers are rejected here: forwarding the
	// mode header would make AP dispatches re-enter the step, and identity
	// headers are step-owned.
	ForwardHeaders []string `json:"forward_headers,omitempty"`

	// WakeupMode selects how wait mode learns a result has landed.
	// "notify": multiplexed keyspace-notification wake-up (requires Redis
	// notify-keyspace-events to include K and l). "poll": periodic
	// non-destructive polling. "auto" (default): notify when the Redis
	// server's config supports it, else poll.
	WakeupMode string `json:"wakeup_mode,omitempty"`

	// Objectives maps tier -> objective names by classification, stamped on
	// passthrough requests as ObjectiveHeader. Tiers without an entry are not
	// stamped. Queued dispatches are stamped by the AP's merge policy
	// (lane_objectives), not here.
	Objectives      map[string]asyncObjectivePair `json:"objectives,omitempty"`
	ObjectiveHeader string                        `json:"objective_header,omitempty"`
	FairnessHeader  string                        `json:"fairness_header,omitempty"`
}

func (c *asyncBrokerConfig) applyDefaults() {
	if c.ModeHeader == "" {
		c.ModeHeader = defaultModeHeader
	}
	if c.TenantHeader == "" {
		c.TenantHeader = "X-Team"
	}
	if c.TimeoutHeader == "" {
		c.TimeoutHeader = "X-Request-Timeout-Seconds"
	}
	if c.DefaultQueue == "" {
		c.DefaultQueue = "request-sortedset"
	}
	if c.Quota.Prefix == "" {
		c.Quota.Prefix = "quota:"
	}
	// Matches the AP's redis-quota gate and fairness stamping defaults, so a
	// default-configured coordinator and AP share quota counters and stamp
	// fairness without explicit attribute alignment.
	if c.Quota.Attribute == "" {
		c.Quota.Attribute = "userid"
	}
	if c.Quota.WindowSeconds == 0 {
		c.Quota.WindowSeconds = 300
	}
	if c.WakeupMode == "" {
		c.WakeupMode = "auto"
	}
	if c.ForwardHeaders == nil {
		c.ForwardHeaders = []string{metadata.TTFTSLOHeaderKey, metadata.TPOTSLOHeaderKey}
	}
	if c.ObjectiveHeader == "" {
		c.ObjectiveHeader = metadata.ObjectiveKey
	}
	if c.FairnessHeader == "" {
		c.FairnessHeader = metadata.FlowFairnessIDKey
	}
}

func (c *asyncBrokerConfig) validate() error {
	if c.RedisURL == "" {
		return errors.New("redis_url is required")
	}
	switch c.WakeupMode {
	case "auto", "notify", "poll":
	default:
		return fmt.Errorf("wakeup_mode must be auto, notify, or poll, got %q", c.WakeupMode)
	}
	if c.FetchGraceSeconds != nil && *c.FetchGraceSeconds < 0 {
		return fmt.Errorf("fetch_grace_seconds must not be negative, got %d", *c.FetchGraceSeconds)
	}
	if c.WaitCapSeconds < 0 {
		return fmt.Errorf("wait_cap_seconds must not be negative, got %d", c.WaitCapSeconds)
	}
	for mode, b := range c.Timeouts {
		if mode != string(asyncModeEnqueue) && mode != string(asyncModeWait) {
			return fmt.Errorf("timeouts keys must be enqueue or wait, got %q", mode)
		}
		if b.DefaultSeconds < 0 || b.MaxSeconds < 0 {
			return fmt.Errorf("timeouts.%s: seconds must not be negative", mode)
		}
		if b.MaxSeconds > 0 && b.DefaultSeconds > b.MaxSeconds {
			return fmt.Errorf("timeouts.%s: default_seconds %d exceeds max_seconds %d", mode, b.DefaultSeconds, b.MaxSeconds)
		}
	}
	if c.Quota.WindowSeconds < 0 {
		return fmt.Errorf("quota.window_seconds must not be negative, got %d", c.Quota.WindowSeconds)
	}
	for tenant, limit := range c.Quota.Limits {
		if limit < 0 {
			return fmt.Errorf("quota.limits.%s must not be negative, got %d", tenant, limit)
		}
	}
	for _, h := range c.ForwardHeaders {
		// The mode header must never ride a queued message: the AP forwards
		// message headers at dispatch, and a forwarded mode header would make
		// the dispatched request re-enter the step. Identity headers are
		// stamped by the step or the AP and cannot be forwarded from clients.
		for _, banned := range []string{c.ModeHeader, c.ObjectiveHeader, c.FairnessHeader} {
			if strings.EqualFold(h, banned) {
				return fmt.Errorf("forward_headers must not include %s", h)
			}
		}
	}
	return nil
}

// fetchGrace resolves the mailbox TTL applied after a delivered fetch, per
// FetchGraceSeconds.
func (c *asyncBrokerConfig) fetchGrace() time.Duration {
	if c.FetchGraceSeconds == nil {
		return resultFetchGraceTTL
	}
	return time.Duration(*c.FetchGraceSeconds) * time.Second
}

// timeoutBounds resolves the deadline bounds for a queued mode, applying the
// fallbacks documented on the Timeouts field. Each field falls back
// independently, so a partial timeouts entry (say, only max_seconds) never
// disturbs the other field's default.
func (c *asyncBrokerConfig) timeoutBounds(m asyncMode) asyncTimeoutBounds {
	b := c.Timeouts[string(m)]
	if b.DefaultSeconds <= 0 {
		if m == asyncModeEnqueue {
			b.DefaultSeconds = defaultEnqueueTimeoutSecs
		} else {
			b.DefaultSeconds = defaultAsyncTimeoutSecs
		}
	}
	if m == asyncModeWait && b.MaxSeconds == 0 {
		b.MaxSeconds = defaultWaitMaxSecs
		if b.DefaultSeconds > b.MaxSeconds {
			b.MaxSeconds = b.DefaultSeconds
		}
	}
	return b
}

// route returns the queue and tier for a (model, tenant) pair.
func (c *asyncBrokerConfig) route(model, tenant string) (queue, tier string) {
	for _, r := range c.Routes {
		if (r.Model == "" || r.Model == model) && (r.Tenant == "" || r.Tenant == tenant) {
			q, t := r.Queue, r.Tier
			if q == "" {
				q = c.DefaultQueue
			}
			if t == "" {
				t = c.DefaultTier
			}
			return q, t
		}
	}
	return c.DefaultQueue, c.DefaultTier
}

// parseAsyncBrokerConfig decodes the step's params block into a validated
// config. Params arrive as the viper-decoded map; the JSON round-trip gives
// typed decoding with unknown-key rejection.
func parseAsyncBrokerConfig(params map[string]any) (*asyncBrokerConfig, error) {
	clean := make(map[string]any, len(params))
	for k, v := range params {
		clean[k] = v
	}
	// The entrypoint injects connector and format defaults into every step's
	// params; they are not async-broker settings.
	delete(clean, ParamKVConnector)
	delete(clean, ParamECConnector)
	delete(clean, "use_openai_format")

	raw, err := json.Marshal(clean)
	if err != nil {
		return nil, fmt.Errorf("invalid params: %w", err)
	}
	var cfg asyncBrokerConfig
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("invalid params: %w", err)
	}
	cfg.applyDefaults()
	if err := cfg.validate(); err != nil {
		return nil, err
	}
	return &cfg, nil
}
