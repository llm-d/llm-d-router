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
	"net/http"
	"net/url"
	"strings"
)

const (
	// PrefillEndpointHeader is the header name used to indicate Prefill worker <ip:port>
	PrefillEndpointHeader = "x-llm-d-prefiller-host-port"

	// EncoderEndpointsHeader is the header name used to indicate Encoder workers <ip:port> list
	EncoderEndpointsHeader = "x-llm-d-encoder-hosts-ports"

	// DataParallelEndpointHeader is the header name used to indicate the worker <ip:port> for Data Parallel.
	// Superseded by native data parallel routing in Istio >= 1.28.1, so it keeps its
	// pre-convention name rather than gaining a second one to remove.
	DataParallelEndpointHeader = "x-data-parallel-host-port"

	// KVCacheSourceHeader is the header name used to indicate the worker <ip:port> holding
	// the most cached prefix KV blocks for the request, to pull from over the P2P connector
	// instead of recomputing them
	KVCacheSourceHeader = "x-llm-d-kv-cache-source-host-port"

	// SpeculativePrefillHeader opts a request into speculative prefill: after the
	// turn completes, the sidecar warms the decoder KV cache with the predicted
	// next-turn prefix (prior messages + assistant answer). Any truthy value
	// (per strconv.ParseBool) enables it; absent or falsey disables it.
	SpeculativePrefillHeader = "x-speculative-prefill"

	// LegacyPrefillEndpointHeader is the pre-convention name of PrefillEndpointHeader.
	LegacyPrefillEndpointHeader = "x-prefiller-host-port"

	// LegacyEncoderEndpointsHeader is the pre-convention name of EncoderEndpointsHeader.
	LegacyEncoderEndpointsHeader = "x-encoder-hosts-ports"

	// LegacyKVCacheSourceHeader is the pre-convention name of KVCacheSourceHeader.
	LegacyKVCacheSourceHeader = "x-kv-cache-source-host-port"

	// InferencePoolAPIGroup is the InferencePool API group
	InferencePoolAPIGroup = "inference.networking.k8s.io"

	// PreferHeader is the standard HTTP "Prefer" header (RFC 7240). EPP
	// receives header keys lowercased.
	PreferHeader = "prefer"

	// PreferIfAvailable is the preference token the coordinator sets to mark a
	// request as a speculative early-decode attempt: route to a decode worker
	// only if its KV cache already covers the prompt (at least partially); otherwise EPP surfaces
	// 412 Precondition Failed so the coordinator restarts the pipeline.
	PreferIfAvailable = "if-available"

	// EndpointPinHeader carries the worker <ip:port> a request must be scheduled
	// on. The endpoint-pin-screener keeps only that endpoint.
	EndpointPinHeader = "x-llm-d-pin-host-port"
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
// Per RFC 7240 the Prefer header value is a comma-separated list of
// preferences. Each preference is a token with an optional "=" value and
// optional ";"-delimited parameters. This function matches the token
// case-insensitively, ignoring whitespace around both the parsed token and
// want, the value, parameters, and any other tokens that may appear alongside
// it. For example, "return=minimal" matches the token "return". A want that is
// empty or only whitespace always returns false. Quoted-string values that
// contain "," or ";" are not supported.
func HasPreference(headers map[string]string, want string) bool {
	prefer := headers[PreferHeader]
	want = strings.TrimSpace(want)
	if prefer == "" || want == "" {
		return false
	}
	for pref := range strings.SplitSeq(prefer, ",") {
		token, _, _ := strings.Cut(pref, ";")
		token, _, _ = strings.Cut(token, "=")
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

// headerAliases maps each disaggregation header to its pre-convention name. EPP
// writes both spellings with the same value, so a sidecar that predates the
// rename still routes. Removing an entry here retires its old name: EPP stops
// writing it and the sidecar starts reading the canonical name (see
// TakeRoutingHeaderValues). Do that once no supported EPP emits the old names.
var headerAliases = map[string]string{
	PrefillEndpointHeader:  LegacyPrefillEndpointHeader,
	EncoderEndpointsHeader: LegacyEncoderEndpointsHeader,
	KVCacheSourceHeader:    LegacyKVCacheSourceHeader,
}

// HeaderNames returns name followed by its deprecated alias, if it has one.
func HeaderNames(name string) []string {
	if alias, ok := headerAliases[name]; ok {
		return []string{name, alias}
	}
	return []string{name}
}

// SetRoutingHeader records value under name and its deprecated alias.
func SetRoutingHeader(headers map[string]string, name, value string) {
	for _, n := range HeaderNames(name) {
		headers[n] = value
	}
}

// DeleteRoutingHeader drops name and its deprecated alias, so a client-supplied
// value cannot survive under either spelling.
func DeleteRoutingHeader(headers map[string]string, name string) {
	for _, n := range HeaderNames(name) {
		delete(headers, n)
	}
}

// trustOrder returns the spellings of name in the order their values are
// trusted: the deprecated alias first, deliberately against the usual
// preference for the current name. Sanitization was deployed under the old
// name, so every supported EPP strips a client-supplied alias on ingress, while
// one that predates the rename forwards the canonical name untouched. Preferring
// the alias keeps a client-supplied canonical value from overriding the target
// such an EPP chose (#3087). An upgraded EPP writes both spellings with the same
// value, so the order changes nothing once EPP is upgraded.
func trustOrder(name string) []string {
	if alias, ok := headerAliases[name]; ok {
		return []string{alias, name}
	}
	return []string{name}
}

// TakeRoutingHeaderValues returns the values of name, preferring its deprecated
// alias (see trustOrder), and removes every spelling from h. Taking and removing
// in one step keeps a request from reaching a worker with a routing header still
// on it.
func TakeRoutingHeaderValues(h http.Header, name string) []string {
	var values []string
	for _, n := range trustOrder(name) {
		if len(values) == 0 {
			values = h.Values(n)
		}
	}

	for _, n := range HeaderNames(name) {
		h.Del(n)
	}
	return values
}

// TakeRoutingHeaderValue is TakeRoutingHeaderValues for a single-valued header.
func TakeRoutingHeaderValue(h http.Header, name string) string {
	values := TakeRoutingHeaderValues(h, name)
	if len(values) == 0 {
		return ""
	}
	return values[0]
}
