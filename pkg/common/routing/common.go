// Package routing contains routing constants and utilities shared between
// the EPP/Inference-Scheduler and the Routing Sidecar.
//
//revive:disable:var-naming
package routing

import "net/url"

const (
	// PrefillEndpointHeader is the header name used to indicate Prefill worker <ip:port>
	PrefillEndpointHeader = "x-prefiller-host-port"

	// EncoderEndpointsHeader is the header name used to indicate Encoder workers <ip:port> list
	EncoderEndpointsHeader = "x-encoder-hosts-ports"

	// DataParallelEndpointHeader is the header name used to indicate the worker <ip:port> for Data Parallel
	DataParallelEndpointHeader = "x-data-parallel-host-port"

	// InferencePoolAPIGroup is the default InferencePool API group
	InferencePoolAPIGroup = "inference.networking.k8s.io"
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
