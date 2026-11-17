package proxy

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"sigs.k8s.io/controller-runtime/pkg/log"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

func TestConcurrentPDStreamingPrefillFailureBeforeResponseCommit(t *testing.T) {
	for _, connector := range []string{constants.KVConnectorSGLang, constants.KVConnectorMooncake} {
		t.Run(connector, func(t *testing.T) {
			decodeHeadersSent := make(chan struct{})
			prefill := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				select {
				case <-decodeHeadersSent:
				case <-r.Context().Done():
					return
				}
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusServiceUnavailable)
				_, _ = w.Write([]byte(`{"object":"error","message":"prefill overloaded","code":503}`))
			}))
			defer prefill.Close()

			decode := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				_, _ = io.Copy(io.Discard, r.Body)
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				w.(http.Flusher).Flush()
				close(decodeHeadersSent)
				<-r.Context().Done()
			}))
			defer decode.Close()

			prefillURL, err := url.Parse(prefill.URL)
			require.NoError(t, err)
			decodeURL, err := url.Parse(decode.URL)
			require.NoError(t, err)

			s := NewProxy(Config{Port: "0", DecoderURL: decodeURL})
			s.logger = log.Log
			s.decoderProxy = s.createDecoderProxyHandler(decodeURL, false)
			sidecar := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				body, _ := io.ReadAll(r.Body)
				s.runConcurrentPD(w, withContextLogger(s, r), body, body, prefillURL.Host, connector, nil, nil)
			}))
			defer sidecar.Close()

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			req, err := http.NewRequestWithContext(ctx, http.MethodPost,
				sidecar.URL+reqcommon.PathChatCompletions,
				strings.NewReader(`{"model":"m","messages":[],"stream":true}`))
			require.NoError(t, err)
			resp, err := sidecar.Client().Do(req)
			require.NoError(t, err)
			defer resp.Body.Close()
			got, readErr := io.ReadAll(resp.Body)
			require.NoError(t, readErr)
			require.Equal(t, http.StatusServiceUnavailable, resp.StatusCode)
			require.Contains(t, string(got), "prefill overloaded")
		})
	}
}
