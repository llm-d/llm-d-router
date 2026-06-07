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

// Package sessionid provides a DataProducer that extracts a session
// identifier from a configured request header or named cookie and tracks
// which endpoint each active session was last routed to. The producer
// publishes two attributes on the InferenceRequest attribute store:
// SessionID (always, when an identifier is present) and BoundEndpoint (when
// the session has been pinned to an endpoint by a previous request). The
// post-schedule PreRequest hook records the chosen endpoint so the next
// request in the same session can be steered back to it.
package sessionid

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	lru "github.com/hashicorp/golang-lru/v2/expirable"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	sessionidconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionid/constants"
)

// SessionIDProducerType is the plugin type registered with the framework.
const SessionIDProducerType = sessionidconstants.SessionIDProducerType

// cookieHeader is the standard HTTP request header carrying cookies.
// Headers in InferenceRequest are normalized to lowercase.
const cookieHeader = "cookie"

const (
	// defaultLRUSize bounds the number of concurrent session bindings tracked.
	defaultLRUSize = 1024

	// defaultTTL is how long a binding survives without activity before
	// eviction. Both PreRequest writes and successful Produce reads refresh
	// the entry, so an active session keeps its binding alive.
	defaultTTL = 30 * time.Minute

	// defaultAutoHeaderName is the source header used when the producer is
	// auto-instantiated as the default for SessionIDDataKey or
	// BoundEndpointDataKey, i.e. invoked without explicit parameters.
	defaultAutoHeaderName = "x-session-id"
)

// Parameters configures the session-id producer.
//
// Source selection: exactly one of HeaderName or CookieName must be set when
// the producer is configured explicitly. When the producer is
// auto-instantiated as the default for SessionIDDataKey or
// BoundEndpointDataKey (i.e. invoked without a parameters block), HeaderName
// defaults to "x-session-id".
//   - HeaderName: read the value of the named request header verbatim.
//   - CookieName: parse the standard "cookie" request header and read the
//     value of the named cookie.
//
// Binding store (optional, applies to the BoundEndpoint attribute):
//   - LRUSize: maximum number of sessions retained, default 1024.
//   - TTL: idle lifetime of a binding as a Go duration ("30m", "1h"),
//     default "30m". Must be > 0 when set.
type Parameters struct {
	HeaderName string `json:"headerName,omitempty"`
	CookieName string `json:"cookieName,omitempty"`
	LRUSize    int    `json:"lruSize,omitempty"`
	TTL        string `json:"ttl,omitempty"`
}

var (
	_ requestcontrol.DataProducer = &Producer{}
	_ requestcontrol.PreRequest   = &Producer{}
)

// Producer extracts a session identifier from each incoming request,
// publishes it as a SessionID attribute, and looks up / records the
// endpoint that the session is bound to.
type Producer struct {
	typedName  fwkplugin.TypedName
	sessionDK  fwkplugin.DataKey
	bindingDK  fwkplugin.DataKey
	headerName string
	cookieName string
	bindings   *lru.LRU[attrsession.SessionID, attrsession.BoundEndpoint]
}

// Factory builds a Producer from raw plugin parameters. When rawParameters
// is nil (auto-instantiation as the default producer for SessionIDDataKey or
// BoundEndpointDataKey), HeaderName defaults to defaultAutoHeaderName so a
// sensible producer comes up without explicit configuration.
func Factory(name string, rawParameters *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := Parameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&params); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' producer: %w", SessionIDProducerType, err)
		}
	} else {
		params.HeaderName = defaultAutoHeaderName
	}

	header := strings.ToLower(strings.TrimSpace(params.HeaderName))
	cookie := strings.TrimSpace(params.CookieName)

	switch {
	case header == "" && cookie == "":
		return nil, fmt.Errorf("'%s' requires exactly one of headerName or cookieName to be set", SessionIDProducerType)
	case header != "" && cookie != "":
		return nil, fmt.Errorf("'%s' requires exactly one of headerName or cookieName to be set, not both", SessionIDProducerType)
	}

	size := defaultLRUSize
	if params.LRUSize != 0 {
		if params.LRUSize < 0 {
			return nil, fmt.Errorf("'%s': lruSize must be > 0, got %d", SessionIDProducerType, params.LRUSize)
		}
		size = params.LRUSize
	}

	ttl := defaultTTL
	if params.TTL != "" {
		parsed, err := time.ParseDuration(params.TTL)
		if err != nil {
			return nil, fmt.Errorf("'%s': invalid ttl %q: %w", SessionIDProducerType, params.TTL, err)
		}
		if parsed <= 0 {
			return nil, fmt.Errorf("'%s': ttl must be > 0, got %s", SessionIDProducerType, parsed)
		}
		ttl = parsed
	}

	return &Producer{
		typedName:  fwkplugin.TypedName{Type: SessionIDProducerType, Name: name},
		sessionDK:  attrsession.SessionIDDataKey.WithNonEmptyProducerName(name),
		bindingDK:  attrsession.BoundEndpointDataKey.WithNonEmptyProducerName(name),
		headerName: header,
		cookieName: cookie,
		bindings:   lru.NewLRU[attrsession.SessionID, attrsession.BoundEndpoint](size, nil, ttl),
	}, nil
}

// TypedName returns the type and name of the plugin.
func (p *Producer) TypedName() fwkplugin.TypedName {
	return p.typedName
}

// Produces declares the SessionID and BoundEndpoint attribute keys this
// producer may write.
func (p *Producer) Produces() map[fwkplugin.DataKey]any {
	return map[fwkplugin.DataKey]any{
		p.sessionDK: attrsession.SessionID(""),
		p.bindingDK: attrsession.BoundEndpoint(""),
	}
}

// Produce extracts the session identifier from the request, publishes it as
// the SessionID attribute, and, if the session is currently bound to an
// endpoint, also publishes BoundEndpoint. Absence of either is a no-op:
// consumers must treat missing attributes as "no preference".
func (p *Producer) Produce(_ context.Context, request *fwksched.InferenceRequest, _ []fwksched.Endpoint) error {
	if request == nil {
		return nil
	}
	id := p.extract(request)
	if id == "" {
		return nil
	}
	sessionID := attrsession.SessionID(id)
	request.PutAttribute(p.sessionDK.String(), sessionID)
	if bound, ok := p.bindings.Get(sessionID); ok {
		// Re-Add to refresh the TTL: an active session that keeps reading
		// its binding should not expire under a write-only refresh policy.
		//
		// Get-then-Add is not atomic, and the expirable LRU exposes no
		// atomic refresh-on-read. A concurrent PreRequest for the same
		// session can interleave between Get and Add, after which the Add
		// here overwrites the freshly-bound endpoint with the value just
		// read. That causes at most one stale read on the next request;
		// the next PreRequest on the same session restores the correct
		// binding. We accept the bounded flap rather than introduce
		// per-session locking for self-healing behavior.
		p.bindings.Add(sessionID, bound)
		request.PutAttribute(p.bindingDK.String(), bound)
	}
	return nil
}

// PreRequest records the endpoint chosen by the scheduler for this session,
// refreshing the entry's TTL on each call. Requests without a session, or
// without a primary-profile target, are ignored.
func (p *Producer) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) {
	logger := log.FromContext(ctx).V(logutil.DEBUG)

	if request == nil || schedulingResult == nil {
		return
	}
	id := p.extract(request)
	if id == "" {
		return
	}
	target, ok := primaryTarget(schedulingResult)
	if !ok {
		logger.Info("session-id-producer: no primary target endpoint to bind", "sessionID", id)
		return
	}
	p.bindings.Add(attrsession.SessionID(id), target)
	logger.Info("session-id-producer: bound session", "sessionID", id, "endpoint", target)
}

func (p *Producer) extract(request *fwksched.InferenceRequest) string {
	if request == nil || request.Headers == nil {
		return ""
	}
	if p.headerName != "" {
		return strings.TrimSpace(request.Headers[p.headerName])
	}
	return strings.TrimSpace(cookieValue(request.Headers[cookieHeader], p.cookieName))
}

// cookieValue returns the value of the named cookie within an HTTP Cookie
// header, or the empty string if the header is empty or the cookie is not
// present. The header is parsed verbatim per RFC 6265 syntax: cookies are
// separated by "; " and each pair is "name=value".
func cookieValue(header, name string) string {
	if header == "" || name == "" {
		return ""
	}
	for pair := range strings.SplitSeq(header, ";") {
		pair = strings.TrimSpace(pair)
		k, v, ok := strings.Cut(pair, "=")
		if !ok {
			continue
		}
		if k == name {
			return v
		}
	}
	return ""
}

// primaryTarget returns the network identity (host:port) of the first
// endpoint chosen by the primary profile, if any. Returns false when the
// result is empty or the chosen endpoint has no Address/Port.
func primaryTarget(result *fwksched.SchedulingResult) (attrsession.BoundEndpoint, bool) {
	if result == nil {
		return "", false
	}
	profile, ok := result.ProfileResults[result.PrimaryProfileName]
	if !ok || profile == nil || len(profile.TargetEndpoints) == 0 {
		return "", false
	}
	form := attrsession.EndpointBoundForm(profile.TargetEndpoints[0])
	if form == "" {
		return "", false
	}
	return form, true
}
