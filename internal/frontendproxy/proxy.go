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

// Package frontendproxy streams HTTP requests to the model frontend.
package frontendproxy

import (
	"errors"
	"log/slog"
	"net/http"
	"net/http/httputil"
	"net/url"
	"time"
)

// New returns a streaming handler with a limit on the incoming request body.
func New(target *url.URL, maxBytes int64, grpc bool) http.Handler {
	transport := &http.Transport{DisableCompression: true, IdleConnTimeout: 90 * time.Second}
	if grpc {
		transport.Protocols = new(http.Protocols)
		transport.Protocols.SetUnencryptedHTTP2(true)
		// Both inference RPCs carry one message with a five-byte gRPC frame header.
		maxBytes += 5
	}
	tooLarge := func(w http.ResponseWriter) {
		if grpc {
			w.Header().Set("Content-Type", "application/grpc")
			w.Header().Set("Grpc-Status", "8")
			w.Header().Set("Grpc-Message", "request body too large")
			w.WriteHeader(http.StatusOK)
			return
		}
		http.Error(w, "request body too large", http.StatusRequestEntityTooLarge)
	}
	proxy := httputil.ReverseProxy{
		Rewrite: func(r *httputil.ProxyRequest) {
			r.SetURL(target)
			r.Out.URL.RawQuery = r.In.URL.RawQuery
		},
		// ReverseProxy flushes SSE and unknown-length streams. An empty gRPC
		// response carries its status in a single terminal HEADERS frame.
		Transport: transport,
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			var sizeErr *http.MaxBytesError
			if errors.As(err, &sizeErr) {
				tooLarge(w)
				return
			}
			if r.Context().Err() == nil {
				slog.ErrorContext(r.Context(), "frontend request failed", "error", err)
			}
			w.WriteHeader(http.StatusBadGateway)
		},
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.ContentLength > maxBytes {
			tooLarge(w)
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, maxBytes)
		proxy.ServeHTTP(w, r) //#nosec G704 -- the frontend origin is configured at startup
	})
}
