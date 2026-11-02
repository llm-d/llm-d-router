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

package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
	"github.com/stretchr/testify/require"
	"k8s.io/utils/set"
)

func TestEPDP2PCacheSource(t *testing.T) {
	for _, ecConnector := range []string{constants.ECConnectorNIXL, constants.ECExampleConnector, constants.ECConnectorMooncake} {
		for _, withPrefiller := range []bool{true, false} {
			path := "encode-decode"
			if withPrefiller {
				path = "encode-prefill-decode"
			}
			for _, scenario := range []string{"enabled", "disabled", "self-source", "malformed", "disallowed", "data-parallel"} {
				if withPrefiller && scenario == "data-parallel" {
					continue
				}
				t.Run(ecConnector+"/"+path+"/"+scenario, func(t *testing.T) {
					newBackend := func(response string) (*httptest.Server, <-chan map[string]any) {
						requests := make(chan map[string]any, 2)
						backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
							var body map[string]any
							if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
								http.Error(w, err.Error(), http.StatusBadRequest)
								return
							}
							select {
							case requests <- body:
							default:
								http.Error(w, "too many backend requests", http.StatusInternalServerError)
								return
							}
							w.Header().Set("Content-Type", "application/json")
							_, _ = io.WriteString(w, response)
						}))
						t.Cleanup(backend.Close)
						return backend, requests
					}
					encoder, encoded := newBackend(`{"ec_transfer_params":{"image-hash":{"peer_host":"10.0.0.1"}}}`)
					prefiller, prefilled := newBackend(`{"kv_transfer_params":{"remote_block_ids":[1],"remote_engine_id":"engine","remote_host":"host","remote_port":4032}}`)
					decoder, decoded := newBackend(`{"choices":[{"message":{"content":"hello"}}]}`)
					decoderURL, err := url.Parse(decoder.URL)
					require.NoError(t, err)
					srv := NewProxy(Config{
						Port: decoderURL.Port(), DecoderURL: decoderURL,
						ECMooncakePodIP: decoderURL.Hostname(),
						KVConnector:     constants.KVConnectorNIXLV2, ECConnector: ecConnector,
						EnableP2PPull: scenario != "disabled", P2PConnectorPort: 7777,
					})
					t.Setenv("POD_IP", decoderURL.Hostname())
					srv.decoderProxy = srv.createDecoderProxyHandler(decoderURL, false)
					prefillHost := strings.TrimPrefix(prefiller.URL, "http://")
					encoderHost := strings.TrimPrefix(encoder.URL, "http://")
					srv.allowlistValidator = &AllowlistValidator{
						enabled:        scenario == "disallowed",
						allowedTargets: set.New(prefillHost, encoderHost),
					}
					proxy := httptest.NewServer(srv.disaggregatedPrefillHandler(reqcommon.APITypeChatCompletions))
					defer proxy.Close()

					source := "10.9.9.9:8000"
					switch scenario {
					case "self-source":
						source = decoderURL.Host
						if withPrefiller {
							source = prefillHost
						}
					case "malformed":
						source = "not-a-host-port"
					}
					body, err := json.Marshal(userMessageRequest(imageURLItem("https://example.com/image.jpg")))
					require.NoError(t, err)
					req, err := http.NewRequest(http.MethodPost, proxy.URL+reqcommon.PathChatCompletions, bytes.NewReader(body))
					require.NoError(t, err)
					req.Header.Set(routing.EncoderEndpointsHeader, encoderHost)
					if withPrefiller {
						req.Header.Set(routing.PrefillEndpointHeader, prefillHost)
					}
					req.Header.Set(routing.KVCacheSourceHeader, source)
					if scenario == "data-parallel" {
						srv.forwardDataParallel = true
						srv.dataParallelProxies[decoderURL.Host] = srv.decoderProxy
						req.Header.Set(routing.DataParallelEndpointHeader, decoderURL.Host)
					}
					client := proxy.Client()
					client.Timeout = 5 * time.Second
					response, err := client.Do(req)
					require.NoError(t, err)
					defer response.Body.Close()
					responseBody, err := io.ReadAll(response.Body)
					require.NoError(t, err)
					require.Equal(t, http.StatusOK, response.StatusCode, string(responseBody))
					require.JSONEq(t, `{"choices":[{"message":{"content":"hello"}}]}`, string(responseBody))
					require.Len(t, encoded, 1)
					require.Len(t, decoded, 1)
					consumerBody := <-decoded
					if withPrefiller {
						require.Len(t, prefilled, 1)
						consumerBody = <-prefilled
					} else {
						require.Empty(t, prefilled)
					}
					if ecConnector == constants.ECConnectorNIXL {
						require.Contains(t, consumerBody[reqcommon.FieldECTransferParams], "image-hash")
					}
					kv, _ := consumerBody[reqcommon.FieldKVTransferParams].(map[string]any)
					if scenario == "enabled" {
						require.Contains(t, kv, requestFieldRemoteKVSource)
						peer := kv[requestFieldRemoteKVSource].(map[string]any)
						require.Equal(t, "10.9.9.9", peer[reqcommon.FieldRemoteHost])
						require.EqualValues(t, 7777, peer[reqcommon.FieldRemotePort])
						require.NotEmpty(t, peer[requestFieldKVRequestID])
					} else {
						require.NotContains(t, kv, requestFieldRemoteKVSource)
					}
				})
			}
		}
	}
}
