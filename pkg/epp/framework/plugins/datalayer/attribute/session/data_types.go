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

// Package session declares the SessionID and BoundEndpoint attributes that
// carry per-request session identity and the endpoint a session is bound to,
// for affinity scoring and filtering. Each attribute is published at most
// once per request on the InferenceRequest attribute store.
package session

import (
	"net"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	sessionidconstants "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionid/constants"
)

// SessionIDDataKey identifies the session identifier published on the request
// attribute store. The default producer is the session-id-producer.
var SessionIDDataKey = plugin.NewDataKey("SessionIDDataKey", sessionidconstants.SessionIDProducerType)

// BoundEndpointDataKey identifies the endpoint currently bound to the
// request's session, published on the request attribute store. The default
// producer is the session-id-producer, which also tracks bindings.
var BoundEndpointDataKey = plugin.NewDataKey("BoundEndpointDataKey", sessionidconstants.SessionIDProducerType)

// SessionID is the session identifier extracted from a request.
type SessionID string

// BoundEndpoint identifies the network destination a session is pinned to.
// It is the canonical "host:port" form returned by net.JoinHostPort over the
// endpoint's Address and Port. Affinity plugins compare this against
// candidate endpoints to enforce stickiness; using the network identity
// (rather than the K8s namespaced name) means a restarted pod with a fresh
// IP no longer matches a stale binding, which is the desired behavior since
// the KV cache the binding existed to preserve died with the previous
// process.
type BoundEndpoint string

// EndpointBoundForm returns the canonical BoundEndpoint value for an
// endpoint, i.e. net.JoinHostPort(Address, Port). It returns the empty
// BoundEndpoint when metadata is missing or either coordinate is empty.
//
// This is the single definition of "the same endpoint" shared by the
// session-id-producer (when recording a binding) and the session-affinity
// scorer/filter (when comparing candidates). Callers that need to test
// equality against a stored BoundEndpoint should use this helper rather
// than constructing the host:port string themselves.
func EndpointBoundForm(ep fwksched.Endpoint) BoundEndpoint {
	if ep == nil {
		return ""
	}
	meta := ep.GetMetadata()
	if meta == nil || meta.Address == "" || meta.Port == "" {
		return ""
	}
	return BoundEndpoint(net.JoinHostPort(meta.Address, meta.Port))
}

// ReadSessionID returns the SessionID published by the default producer on the
// request attribute store, or "" and false if absent.
//
// Consumers should use this helper rather than reading the attribute directly:
// it encapsulates both the key construction and the type assertion, so a
// future change of storage location or value type does not ripple through
// every reader.
func ReadSessionID(r *fwksched.InferenceRequest) (SessionID, bool) {
	key := SessionIDDataKey.WithNonEmptyProducerName(sessionidconstants.SessionIDProducerType).String()
	return fwksched.ReadRequestAttribute[SessionID](r, key)
}
