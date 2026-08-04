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

package sessionaffinity

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// DefaultSessionIDHeader is the header a SessionIDConfig reads when no sources
// are configured.
const DefaultSessionIDHeader = "x-session-id"

// SessionIDConfig configures the session_id strategy.
type SessionIDConfig struct {
	// Sources resolve the session identifier in priority order; the first
	// non-empty match wins. When empty, a single header source
	// DefaultSessionIDHeader is used.
	Sources []SessionIDSource `json:"sources"`
	// EvictionTTLSeconds is how long a session binding survives unused.
	EvictionTTLSeconds float64 `json:"evictionTtlSeconds"`
	// EvictionSweepSeconds is how often expired bindings are swept.
	EvictionSweepSeconds float64 `json:"evictionSweepSeconds"`
}

// SessionIDSource names one place a session identifier can be read from.
// Exactly one of Header or Attribute must be set.
type SessionIDSource struct {
	// Header is a request header name.
	Header string `json:"header"`
	// Attribute is a request-attribute key published by an upstream plugin.
	Attribute string `json:"attribute"`
}

// Default eviction schedule applied to a SessionIDConfig with unset fields:
// bindings unused for 300s are swept every 10s.
const (
	defaultEvictionTTLSeconds   = 300
	defaultEvictionSweepSeconds = 10
)

// ApplyDefaults fills unset fields with their defaults: an empty Sources reads
// DefaultSessionIDHeader, and a zero eviction schedule uses the standard TTL
// and sweep. Run after decode and before Validate. Defaults live here rather
// than in a pre-seeded struct so a JSON overlay replaces Sources instead of
// merging element-wise into a seeded slice.
func (c *SessionIDConfig) ApplyDefaults() {
	if len(c.Sources) == 0 {
		c.Sources = []SessionIDSource{{Header: DefaultSessionIDHeader}}
	}
	if c.EvictionTTLSeconds == 0 {
		c.EvictionTTLSeconds = defaultEvictionTTLSeconds
	}
	if c.EvictionSweepSeconds == 0 {
		c.EvictionSweepSeconds = defaultEvictionSweepSeconds
	}
}

// Validate rejects configs that would not select a valid session identifier or
// a usable eviction schedule. It expects a config already passed through
// applyDefaults.
func (c *SessionIDConfig) Validate() error {
	for i, source := range c.Sources {
		if err := source.validate(); err != nil {
			return fmt.Errorf("sources[%d]: %w", i, err)
		}
	}
	if c.EvictionTTLSeconds <= 0 {
		return fmt.Errorf("evictionTtlSeconds must be > 0, got %v", c.EvictionTTLSeconds)
	}
	if c.EvictionSweepSeconds <= 0 {
		return fmt.Errorf("evictionSweepSeconds must be > 0, got %v", c.EvictionSweepSeconds)
	}
	return nil
}

func (s SessionIDSource) validate() error {
	if (s.Header == "") == (s.Attribute == "") {
		return errors.New("exactly one of header or attribute must be set")
	}
	return nil
}

// SessionIDHeader maps an opaque client-supplied session identifier to a pod
// via a bounded, TTL-evicted binding store. It does not place sessions: an
// unbound session, or one whose bound pod is absent from the candidate set,
// yields no preference, and the consumer decides where the request goes. A
// session whose bound pod is absent rebinds to the pod the request was routed
// to.
//
// It is the shared implementation behind the session-affinity scorer and
// filter: Choose resolves the target endpoint, PreRequest commits the binding,
// ResponseHeader is a no-op. Each plugin embeds it and adds only its own output
// method.
type SessionIDHeader struct {
	sources     []SessionIDSource
	profileName string
	// pluginKey keys the Choose->PreRequest presence handoff and labels the
	// plugin in logs; the owning scorer/filter passes its own plugin type.
	pluginKey     plugin.StateKey
	ttl           time.Duration
	sweepInterval time.Duration
	// mu serializes each bind/migrate/evict sequence so a first-bind cannot race
	// a concurrent sweep and leave the binding store inconsistent.
	mu          sync.Mutex
	bindings    sync.Map // session identifier -> binding
	pluginState *plugin.PluginState
}

// boundPodPresence records, for one request, whether the session's bound pod
// was present in the candidate set scored. PreRequest reads this so it can
// tell a genuine absence from the picker merely choosing a different pod.
type boundPodPresence struct {
	present bool
}

func (b *boundPodPresence) Clone() plugin.StateData {
	return &boundPodPresence{present: b.present}
}

// NewSessionIDHeader builds the shared session_id strategy implementation and
// starts its eviction sweep for the plugin's lifetime. pluginKey is the owning
// plugin's type, used for the PluginState handoff key and log labels.
func NewSessionIDHeader(config SessionIDConfig, profileName string, pluginKey plugin.StateKey, handle plugin.Handle) *SessionIDHeader {
	s := &SessionIDHeader{
		sources:       config.Sources,
		profileName:   profileName,
		pluginKey:     pluginKey,
		ttl:           time.Duration(config.EvictionTTLSeconds * float64(time.Second)),
		sweepInterval: time.Duration(config.EvictionSweepSeconds * float64(time.Second)),
		pluginState:   plugin.NewPluginState(handle.Context()),
	}
	go s.runEviction(handle.Context(), s.sweepInterval)
	return s
}

// binding is the pod a session is pinned to, and when that pin was last used.
type binding struct {
	podName  string
	lastSeen time.Time
}

// resolveSessionID returns the session identifier from the first configured
// source that yields a non-empty value, or "" when none do.
//
// Empty means the request expresses no session preference. No identifier is
// generated: a router-minted value is unique per request, so it could never
// match a stored binding.
func (s *SessionIDHeader) resolveSessionID(ctx context.Context, request *scheduling.InferenceRequest) string {
	if request == nil {
		return ""
	}
	for _, source := range s.sources {
		if source.Header != "" {
			if id := strings.TrimSpace(request.Headers[NormalizeHeader(source.Header)]); id != "" {
				return id
			}
			continue
		}
		if id, ok := attributeString(ctx, request, source.Attribute); ok {
			if id = strings.TrimSpace(id); id != "" {
				return id
			}
		}
	}
	return ""
}

// attributeString reads a request attribute as a string. Producers store their
// identifier under a named string type (e.g. session.SessionID), so a plain
// string type assertion would miss them; any value of string kind is accepted.
func attributeString(ctx context.Context, request *scheduling.InferenceRequest, key string) (string, bool) {
	v, ok := request.GetAttribute(key)
	if !ok {
		return "", false
	}
	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.String {
		log.FromContext(ctx).V(logging.DEBUG).Info("Session affinity - attribute is not a string, skipping source",
			"attribute", key)
		return "", false
	}
	return rv.String(), true
}

// Choose resolves the endpoint a request should be pinned to and records, for
// PreRequest, whether the session's bound pod was present in the candidate set.
// It returns (nil, false) when there are no candidates, the request carries no
// session identifier, the session is unbound, or its bound pod is absent from
// the candidate set: in each case the session expresses no usable preference
// and the consumer decides. It returns (pod, true) only for a session bound to
// a present candidate. Choose does not mutate the store; the binding is
// committed in PreRequest.
func (s *SessionIDHeader) Choose(ctx context.Context, request *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) (scheduling.Endpoint, bool) {
	if len(endpoints) == 0 {
		return nil, false
	}

	sessionID := s.resolveSessionID(ctx, request)
	if sessionID == "" {
		return nil, false
	}

	podName := s.boundPod(sessionID)
	var target scheduling.Endpoint
	for _, endpoint := range endpoints {
		if podName != "" && endpoint.GetMetadata().ID.String() == podName {
			target = endpoint
		}
	}

	if podName != "" {
		s.pluginState.Write(request.RequestID, s.pluginKey, &boundPodPresence{present: target != nil})
	}

	if target == nil {
		return nil, false
	}
	return target, true
}

// boundPod returns the pod bound to sessionID, or "" when unbound. Expiry is
// enforced by the sweeper, not on read.
func (s *SessionIDHeader) boundPod(sessionID string) string {
	if sessionID == "" {
		return ""
	}
	value, ok := s.bindings.Load(sessionID)
	if !ok {
		return ""
	}
	b, ok := value.(binding)
	if !ok {
		return ""
	}
	return b.podName
}

// PreRequest is the only place bindings are mutated. A session migrates only
// when its bound pod is absent from the candidate set.
func (s *SessionIDHeader) PreRequest(ctx context.Context, request *scheduling.InferenceRequest, schedulingResult *scheduling.SchedulingResult) {
	sessionID := s.resolveSessionID(ctx, request)
	if sessionID == "" {
		return
	}
	podName := s.pickedPodName(schedulingResult)
	if podName == "" {
		return
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	fresh := binding{podName: podName, lastSeen: time.Now()}
	existingValue, loaded := s.bindings.LoadOrStore(sessionID, fresh)
	if !loaded {
		log.FromContext(ctx).V(logging.DEBUG).Info("Session affinity - bound session to pod",
			"plugin", string(s.pluginKey), "pod", podName)
		return
	}

	existing, ok := existingValue.(binding)
	if !ok {
		return
	}

	if existing.podName == podName {
		s.bindings.CompareAndSwap(sessionID, existingValue, fresh)
		return
	}

	// Migrate only on a confirmed absence recorded by this request's own Choose
	// call. A missing record (e.g. a concurrent first-bind race) is not treated
	// as absence, so it never forces a migration.
	present, err := plugin.ReadPluginStateKey[*boundPodPresence](s.pluginState, request.RequestID, s.pluginKey)
	if err != nil || present == nil || present.present {
		return
	}

	if !s.bindings.CompareAndSwap(sessionID, existingValue, fresh) {
		return
	}
	log.FromContext(ctx).V(logging.DEBUG).Info("Session affinity - session migrated pod",
		"plugin", string(s.pluginKey), "from", existing.podName, "to", podName)
}

// pickedPodName returns the pod chosen for this instance's profile, or "" when
// that profile was not scheduled. An empty profileName selects the primary
// (decode) profile.
func (s *SessionIDHeader) pickedPodName(schedulingResult *scheduling.SchedulingResult) string {
	if schedulingResult == nil {
		return ""
	}
	profileName := s.profileName
	if profileName == "" {
		profileName = schedulingResult.PrimaryProfileName
	}
	result, ok := schedulingResult.ProfileResults[profileName]
	if !ok || result == nil || len(result.TargetEndpoints) == 0 || result.TargetEndpoints[0] == nil {
		return ""
	}
	if md := result.TargetEndpoints[0].GetMetadata(); md != nil {
		return md.ID.String()
	}
	return ""
}

// ResponseHeader is a no-op: under session_id the client owns its
// session identifier.
func (s *SessionIDHeader) ResponseHeader(context.Context, *scheduling.InferenceRequest, *requestcontrol.Response, *datalayer.EndpointMetadata) {
}

// expired reports whether a binding last used at lastSeen has outlived the TTL.
func (s *SessionIDHeader) expired(lastSeen, now time.Time) bool {
	return s.ttl > 0 && now.Sub(lastSeen) > s.ttl
}

// runEviction drops bindings unused for longer than the TTL until ctx is cancelled.
func (s *SessionIDHeader) runEviction(ctx context.Context, interval time.Duration) {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			now := time.Now()
			s.bindings.Range(func(key, value any) bool {
				b, ok := value.(binding)
				if !ok {
					s.mu.Lock()
					s.bindings.CompareAndDelete(key, value)
					s.mu.Unlock()
					return true
				}
				if !s.expired(b.lastSeen, now) {
					return true
				}
				// Re-check under the lock: a concurrent preRequest may have
				// refreshed this binding since Range snapshotted it, in which
				// case CompareAndDelete correctly declines to remove it.
				s.mu.Lock()
				s.bindings.CompareAndDelete(key, value)
				s.mu.Unlock()
				return true
			})
		}
	}
}
