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

// IsConditionalDecode reports whether the request headers carry the standard
// HTTP "Prefer: if-available" preference (RFC 7240). It is the marker the
// coordinator uses to flag a speculative early-decode attempt: route to a
// decode worker only if its KV cache already covers the prompt(at least partially);
// otherwise EPP surfaces 412 Precondition Failed so the coordinator restarts the
// pipeline.
//
// Per RFC 7240 the Prefer header value is a comma-separated list of preference
// tokens, each with optional ";"-delimited parameters. This function matches
// the bare "if-available" token case-insensitively, ignoring surrounding
// whitespace, parameters, and any other tokens that may appear alongside it.
//
// Headers are expected to be lowercased keys (EPP receives them that way from
// ext-proc; see pkg/epp/handlers/request.go).
func IsConditionalDecode(headers map[string]string) bool {
	for _, pref := range strings.Split(headers[PreferHeader], ",") {
		token, _, _ := strings.Cut(pref, ";")
		if strings.EqualFold(strings.TrimSpace(token), PreferIfAvailable) {
			return true
		}
	}
	return false
}
