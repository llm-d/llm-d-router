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

package proxy

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"k8s.io/utils/set"

	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

// testPrefillHeaderRouting is a shared table-driven helper that exercises
// prefill-header parsing, sampling, passthrough, and P/D protocol invocation
// for any APIType.  Both TestServer_chatCompletionsHandler and
// TestServer_responsesHandler delegate to it.
func testPrefillHeaderRouting(t *testing.T, apiType APIType) {
	t.Helper()
	tests := []struct {
		name     string
		sampling bool
		r        *http.Request

		expectedCode             int
		expectedPrefillHostPorts []string
		expectedPassthrough      bool
	}{
		{
			name: "passthrough by default",
			r:    &http.Request{},

			expectedPassthrough: true,
		},
		{
			name: "passthrough with no header value",
			r:    &http.Request{Header: http.Header{http.CanonicalHeaderKey(routing.PrefillEndpointHeader): []string{}}},

			expectedPassthrough: true,
		},
		{
			name: "default prefill to one header value",
			r:    &http.Request{Header: http.Header{http.CanonicalHeaderKey(routing.PrefillEndpointHeader): []string{"a"}}},

			expectedCode:             200,
			expectedPrefillHostPorts: []string{"a"},
		},
		{
			name: "default prefill to first header value",
			r:    &http.Request{Header: http.Header{http.CanonicalHeaderKey(routing.PrefillEndpointHeader): []string{"a,b"}}},

			expectedCode:             200,
			expectedPrefillHostPorts: []string{"a"},
		},
		{
			name:     "sample from comma delimited header",
			r:        &http.Request{Header: http.Header{http.CanonicalHeaderKey(routing.PrefillEndpointHeader): []string{"a,b"}}},
			sampling: true,

			expectedCode:             200,
			expectedPrefillHostPorts: []string{"a", "b"},
		},
		{
			name:     "sample from comma delimited header with whitespace",
			r:        &http.Request{Header: http.Header{http.CanonicalHeaderKey(routing.PrefillEndpointHeader): []string{" a, b"}}},
			sampling: true,

			expectedCode:             200,
			expectedPrefillHostPorts: []string{"a", "b"},
		},
		{
			name:     "sample from duplicate values",
			r:        &http.Request{Header: http.Header{http.CanonicalHeaderKey(routing.PrefillEndpointHeader): []string{"a,a"}}},
			sampling: true,

			expectedCode:             200,
			expectedPrefillHostPorts: []string{"a"},
		},
		{
			name:     "sample from multiple header values",
			r:        &http.Request{Header: http.Header{http.CanonicalHeaderKey(routing.PrefillEndpointHeader): []string{"a", "b"}}},
			sampling: true,

			expectedCode:             200,
			expectedPrefillHostPorts: []string{"a", "b"},
		},
		{
			name:     "sample from empty header value",
			r:        &http.Request{Header: http.Header{http.CanonicalHeaderKey(routing.PrefillEndpointHeader): []string{""}}},
			sampling: true,

			expectedPassthrough: true,
		},
		{
			name:     "sample from multiple empty header values",
			r:        &http.Request{Header: http.Header{http.CanonicalHeaderKey(routing.PrefillEndpointHeader): []string{"", ""}}},
			sampling: true,

			expectedPassthrough: true,
		},
	}
	for _, tt := range tests {
		maxAttempts := len(tt.expectedPrefillHostPorts) + 1

		for i := 0; i < maxAttempts; i++ {
			t.Run(fmt.Sprintf("%s_%d", tt.name, i), func(t *testing.T) {
				s := NewProxy(Config{Port: "8000", EnablePrefillerSampling: tt.sampling})
				s.allowlistValidator = &AllowlistValidator{}
				s.prefillSamplerFn = func(n int) int { return i % n }
				var hostPort string
				var capturedReq *http.Request
				s.runPDConnectorProtocol = func(_ http.ResponseWriter, r *http.Request, selectedHostPort string, _ APIType) {
					hostPort = selectedHostPort
					capturedReq = r
				}
				var passthrough bool
				s.decoderProxy = http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
					passthrough = true
					capturedReq = r
				})
				s.dataParallelProxies = make(map[string]http.Handler)
				recorder := httptest.NewRecorder()
				recorder.Code = 0
				req := tt.r.Clone(tt.r.Context())
				s.disaggregatedPrefillHandler(apiType)(recorder, req)

				resp := recorder.Result()
				if passthrough {
					if !tt.expectedPassthrough {
						t.Errorf("unexpected passthrough to decode")
					}
					if recorder.Code != 0 || recorder.Body.Len() > 0 || len(resp.Header) > 0 {
						t.Errorf("unexpected write to recorder during passthrough: %#v %#v", recorder, resp)
					}
					if len(hostPort) > 0 {
						t.Errorf("unexpected hostPort set")
					}
				} else {
					if tt.expectedPassthrough {
						t.Fatal("unexpected handled request")
					}
					if resp.StatusCode != tt.expectedCode {
						t.Errorf("unexpected code: %d", resp.StatusCode)
					}
					expected, actual := tt.expectedPrefillHostPorts[i%len(tt.expectedPrefillHostPorts)], hostPort
					if expected != actual {
						t.Errorf("expected=%s actual=%s", expected, actual)
					}
				}
				if capturedReq != nil {
					if v := capturedReq.Header.Get(routing.PrefillEndpointHeader); v != "" {
						t.Errorf("PrefillEndpointHeader should be stripped before forwarding, got %q", v)
					}
				}
			})
		}
	}
}

func TestServer_chatCompletionsHandler(t *testing.T) {
	testPrefillHeaderRouting(t, APITypeChatCompletions)
}

func TestServer_responsesHandler(t *testing.T) {
	testPrefillHeaderRouting(t, APITypeResponses)
}

func TestServer_encoderEndpointRouting(t *testing.T) {
	encoderHeader := http.CanonicalHeaderKey(routing.EncoderEndpointsHeader)
	prefillHeader := http.CanonicalHeaderKey(routing.PrefillEndpointHeader)

	tests := []struct {
		name string
		r    *http.Request

		// allowlist config: nil means disabled (allow all), non-nil means enabled with given hosts
		allowedHosts []string

		// set to true to install a mock EPD runner
		epdConfigured bool

		expectedEPD         bool
		expectedEPDEncoders []string
		expectedEPDPrefill  string
		expectedPD          bool
		expectedPDHost      string
		expectedPassthrough bool
	}{
		{
			name: "encoder header allowed invokes EPD protocol",
			r: &http.Request{Header: http.Header{
				encoderHeader: []string{"enc1:8000"},
			}},
			epdConfigured:       true,
			expectedEPD:         true,
			expectedEPDEncoders: []string{"enc1:8000"},
			expectedEPDPrefill:  "",
		},
		{
			name: "multiple encoder headers allowed invokes EPD protocol with all",
			r: &http.Request{Header: http.Header{
				encoderHeader: []string{"enc1:8000,enc2:8000"},
			}},
			epdConfigured:       true,
			expectedEPD:         true,
			expectedEPDEncoders: []string{"enc1:8000", "enc2:8000"},
			expectedEPDPrefill:  "",
		},
		{
			name: "encoder header all denied falls back to decoder passthrough",
			r: &http.Request{
				Header: http.Header{
					encoderHeader: []string{"enc1:8000"},
				},
				URL: &url.URL{Path: "/v1/chat/completions"},
			},
			allowedHosts:        []string{"other-host"},
			epdConfigured:       true,
			expectedPassthrough: true,
		},
		{
			name: "encoder header partially denied filters to allowed only",
			r: &http.Request{
				Header: http.Header{
					encoderHeader: []string{"enc1:8000,denied:8000"},
				},
				URL: &url.URL{Path: "/v1/chat/completions"},
			},
			allowedHosts:        []string{"enc1"},
			epdConfigured:       true,
			expectedEPD:         true,
			expectedEPDEncoders: []string{"enc1:8000"},
			expectedEPDPrefill:  "",
		},
		{
			name: "encoder header all denied falls back to P/D when prefill header present",
			r: &http.Request{
				Header: http.Header{
					encoderHeader: []string{"enc1:8000"},
					prefillHeader: []string{"prefill1:8000"},
				},
				URL: &url.URL{Path: "/v1/chat/completions"},
			},
			allowedHosts:   []string{"prefill1"},
			epdConfigured:  true,
			expectedPD:     true,
			expectedPDHost: "prefill1:8000",
		},
		{
			name: "encoder and prefill headers both allowed invokes EPD with prefill host",
			r: &http.Request{Header: http.Header{
				encoderHeader: []string{"enc1:8000"},
				prefillHeader: []string{"prefill1:8000"},
			}},
			epdConfigured:       true,
			expectedEPD:         true,
			expectedEPDEncoders: []string{"enc1:8000"},
			expectedEPDPrefill:  "prefill1:8000",
		},
		{
			name: "encoder header allowed but no EPD connector configured falls back to decoder",
			r: &http.Request{Header: http.Header{
				encoderHeader: []string{"enc1:8000"},
			}},
			epdConfigured:       false,
			expectedPassthrough: true,
		},
		{
			name: "encoder header allowed but no EPD connector with prefill header falls back to P/D",
			r: &http.Request{Header: http.Header{
				encoderHeader: []string{"enc1:8000"},
				prefillHeader: []string{"prefill1:8000"},
			}},
			epdConfigured:  false,
			expectedPD:     true,
			expectedPDHost: "prefill1:8000",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s := NewProxy(Config{Port: "8000"})

			if tt.allowedHosts != nil {
				s.allowlistValidator = &AllowlistValidator{
					enabled:        true,
					allowedTargets: set.New(tt.allowedHosts...),
				}
			} else {
				s.allowlistValidator = &AllowlistValidator{}
			}

			var epdCalled bool
			var epdPrefill string
			var epdEncoders []string
			var capturedReq *http.Request
			if tt.epdConfigured {
				s.runEPDConnectorProtocol = func(_ http.ResponseWriter, r *http.Request, prefillHost string, encoders []string) {
					epdCalled = true
					epdPrefill = prefillHost
					epdEncoders = encoders
					capturedReq = r
				}
			}

			var pdCalled bool
			var pdHost string
			s.runPDConnectorProtocol = func(_ http.ResponseWriter, r *http.Request, host string, _ APIType) {
				pdCalled = true
				pdHost = host
				capturedReq = r
			}

			var passthrough bool
			s.decoderProxy = http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
				passthrough = true
				capturedReq = r
			})
			s.dataParallelProxies = make(map[string]http.Handler)

			recorder := httptest.NewRecorder()
			recorder.Code = 0
			s.disaggregatedPrefillHandler(APITypeChatCompletions)(recorder, tt.r)

			switch {
			case tt.expectedEPD:
				if !epdCalled {
					t.Fatal("expected EPD protocol to be called, but it was not")
				}
				if epdPrefill != tt.expectedEPDPrefill {
					t.Errorf("EPD prefill host: got %q, want %q", epdPrefill, tt.expectedEPDPrefill)
				}
				if len(epdEncoders) != len(tt.expectedEPDEncoders) {
					t.Fatalf("EPD encoders: got %v, want %v", epdEncoders, tt.expectedEPDEncoders)
				}
				for i, enc := range tt.expectedEPDEncoders {
					if epdEncoders[i] != enc {
						t.Errorf("EPD encoder[%d]: got %q, want %q", i, epdEncoders[i], enc)
					}
				}
				if pdCalled {
					t.Error("P/D protocol should not be called when EPD is used")
				}
				if passthrough {
					t.Error("decoder passthrough should not happen when EPD is used")
				}
			case tt.expectedPD:
				if !pdCalled {
					t.Fatal("expected P/D protocol to be called, but it was not")
				}
				if pdHost != tt.expectedPDHost {
					t.Errorf("P/D host: got %q, want %q", pdHost, tt.expectedPDHost)
				}
				if epdCalled {
					t.Error("EPD protocol should not be called when falling back to P/D")
				}
				if passthrough {
					t.Error("decoder passthrough should not happen when P/D is used")
				}
			case tt.expectedPassthrough:
				if !passthrough {
					t.Fatal("expected decoder passthrough, but it did not happen")
				}
				if epdCalled {
					t.Error("EPD protocol should not be called during passthrough")
				}
				if pdCalled {
					t.Error("P/D protocol should not be called during passthrough")
				}

			}
			if capturedReq != nil {
				if v := capturedReq.Header.Get(routing.PrefillEndpointHeader); v != "" {
					t.Errorf("PrefillEndpointHeader should be stripped before forwarding, got %q", v)
				}
				if v := capturedReq.Header.Get(routing.EncoderEndpointsHeader); v != "" {
					t.Errorf("EncoderEndpointsHeader should be stripped before forwarding, got %q", v)
				}
			}
		})
	}
}

func TestAPIType_String(t *testing.T) {
	t.Parallel()
	if g, w := APITypeChatCompletions.String(), "chat_completions"; g != w {
		t.Errorf("APITypeChatCompletions.String() = %q, want %q", g, w)
	}
	if g, w := APITypeResponses.String(), "responses"; g != w {
		t.Errorf("APITypeResponses.String() = %q, want %q", g, w)
	}
	if g, w := APIType(7).String(), fmt.Sprintf("APIType(%d)", 7); g != w {
		t.Errorf("APIType(7).String() = %q, want %q", g, w)
	}
}
