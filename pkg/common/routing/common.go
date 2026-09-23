/*
Copyright 2025 The llm-d Authors.

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

// Package routing contains routing constants and utilities shared between
// the EPP/Inference-Scheduler and the Routing Sidecar.
//
//revive:disable:var-naming
package routing

import (
	"net/url"
	"strings"
)

const (
	// PrefillEndpointHeader is the header name used to indicate Prefill worker <ip:port>
	PrefillEndpointHeader = "x-prefiller-host-port"

	// EncoderEndpointsHeader is the header name used to indicate Encoder workers <ip:port> list
	EncoderEndpointsHeader = "x-encoder-hosts-ports"

	// DataParallelEndpointHeader is the header name used to indicate the worker <ip:port> for Data Parallel
	DataParallelEndpointHeader = "x-data-parallel-host-port"

	// KVCacheSourceHeader is the header name used to indicate the worker <ip:port> holding
	// the most cached prefix KV blocks for the request, to pull from over the P2P connector
	// instead of recomputing them
	KVCacheSourceHeader = "x-kv-cache-source-host-port"

	// InferencePoolAPIGroup is the default InferencePool API group
	InferencePoolAPIGroup = "inference.networking.k8s.io"

	// PreferHeader is the standard HTTP "Prefer" header (RFC 7240). EPP
	// receives header keys lowercased.
	PreferHeader = "prefer"

	// PreferIfAvailable is the preference token the coordinator sets to mark a
	// request as a speculative early-decode attempt: route to a decode worker
	// only if its KV cache already covers the prompt (at least partially); otherwise EPP surfaces
	// 412 Precondition Failed so the coordinator restarts the pipeline.
	PreferIfAvailable = "if-available"

	// PreferReserveEndpoint is the preference token the coordinator sets to ask
	// which endpoint EPP would pick for a request without sending the request
	// there. EPP answers 200 with the picked <ip:port> on ReservedEndpointHeader
	// and forwards nothing.
	PreferReserveEndpoint = "reserve-endpoint"

	// ReservedEndpointHeader is the response header that carries the <ip:port>
	// EPP picked on a "Prefer: reserve-endpoint" request. The name is fixed, not
	// derived from the scheduling profile, because a per-phase EPP runs its only
	// profile under another name.
	ReservedEndpointHeader = "x-prefill-host-port"

	// PrefillPinHeader carries the prefill worker <ip:port> a request must be
	// scheduled on. The coordinator copies it from the reserve-endpoint answer
	// so that the prefill request lands on the pod its peer request names.
	PrefillPinHeader = "x-prefill-pin"
)

// StripScheme removes the scheme from an endpoint URL, returning host:port.
// This is useful for gRPC clients that expect host:port format only.
func StripScheme(endpoint string) string {
	u, err := url.Parse(endpoint)
	if err != nil || u.Host == "" {
		return endpoint // not a valid URL, return as-is
	}
	return u.Host
}

// HasPreference reports whether the request headers carry the given Prefer
// token.
//
// Per RFC 7240 the Prefer header value is a comma-separated list of preference
// tokens, each with optional ";"-delimited parameters. This function matches
// the bare token case-insensitively, ignoring surrounding whitespace,
// parameters, and any other tokens that may appear alongside it.
func HasPreference(headers map[string]string, want string) bool {
	prefer := headers[PreferHeader]
	if prefer == "" {
		return false
	}
	for _, pref := range strings.Split(prefer, ",") {
		token, _, _ := strings.Cut(pref, ";")
		if strings.EqualFold(strings.TrimSpace(token), want) {
			return true
		}
	}
	return false
}

// IsConditionalDecode reports whether the request headers carry the
// "Prefer: if-available" preference (see PreferIfAvailable for semantics).
func IsConditionalDecode(headers map[string]string) bool {
	return HasPreference(headers, PreferIfAvailable)
}
