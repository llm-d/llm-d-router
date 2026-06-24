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

package gateway

import (
	"net/http"
	"testing"

	"github.com/llm-d/coordinator/pkg/config"
)

// The gateway reaches in-cluster destinations (Envoy, EPP, model-serving pods).
// Its client must NOT route through HTTP_PROXY/HTTPS_PROXY: it builds an explicit
// http.Transport and leaves the Proxy field nil ("never proxy"). This is the
// opposite of the multimedia downloader, which relies on http.DefaultTransport to
// honor the proxy env. This test fails if someone adds a Proxy to the transport,
// which would send in-cluster traffic through an external forward proxy.
func TestClient_IgnoresProxyEnv(t *testing.T) {
	c := New(config.GatewayConfig{Address: "http://gw"})

	tr, ok := c.httpClient.Transport.(*http.Transport)
	if !ok {
		t.Fatalf("expected *http.Transport, got %T", c.httpClient.Transport)
	}
	if tr.Proxy != nil {
		t.Fatal("gateway transport must not set Proxy: in-cluster traffic must not route through HTTP(S)_PROXY")
	}
}
