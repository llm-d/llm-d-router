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
