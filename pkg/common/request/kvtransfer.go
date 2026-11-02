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

package request

// RemoteDecoderParams returns the kv_transfer_params of an OffloadingConnector
// P2P tier prefill request: the prefill pod stores the KV under kvRequestID
// for a decode pod to pull.
func RemoteDecoderParams(kvRequestID string) map[string]any {
	return map[string]any{
		FieldRemoteDecoder: map[string]any{FieldKVRequestID: kvRequestID},
	}
}

// RemotePrefillerParams returns the kv_transfer_params of an
// OffloadingConnector P2P tier decode request: the decode pod pulls the KV
// stored under kvRequestID from the prefill pod's tier at host:port.
func RemotePrefillerParams(kvRequestID, host string, port int) map[string]any {
	return map[string]any{
		FieldRemotePrefiller: P2PPeerParams(kvRequestID, host, port),
	}
}

// P2PPeerParams returns an OffloadingConnector P2P tier peer block of
// kv_transfer_params: the kv_request_id of the transfer and the host and tier
// port a pull dials. It is the value of remote_prefiller on a decode request
// and of remote_kv_source on a request that pulls a cached prefix.
func P2PPeerParams(kvRequestID, host string, port int) map[string]any {
	return map[string]any{
		FieldKVRequestID: kvRequestID,
		FieldRemoteHost:  host,
		FieldRemotePort:  port,
	}
}
