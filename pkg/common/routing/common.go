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

	// DefaultPoolGroup is the default InferencePool API group
	DefaultPoolGroup = "inference.networking.k8s.io"

	// LegacyPoolGroup is the legacy InferencePool API group
	LegacyPoolGroup = "inference.networking.x-k8s.io"

	// KVConnectorNIXLV2 enables the P/D KV NIXL v2 protocol
	KVConnectorNIXLV2 = "nixlv2"

	// KVConnectorSharedStorage enables the P/D KV Shared Storage protocol
	KVConnectorSharedStorage = "shared-storage"

	// KVConnectorSGLang enables SGLang the P/D KV disaggregation protocol
	KVConnectorSGLang = "sglang"

	// ECExampleConnector enables the Encoder disaggregation protocol (E/PD, E/P/D)
	ECExampleConnector = "ec-example"
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
