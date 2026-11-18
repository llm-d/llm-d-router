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

// Package frontendproxy streams HTTP and gRPC requests to the model frontend.
package frontendproxy

import (
	"errors"
	"net/http"
	"net/http/httputil"
	"net/url"
	"time"

	"google.golang.org/grpc/codes"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
)

// NewHTTP forwards HTTP requests with bodies up to maxBytes.
func NewHTTP(addr string, maxBytes int64) http.Handler {
	return newProxy(addr, maxBytes, nil, func(w http.ResponseWriter) {
		http.Error(w, "request body too large", http.StatusRequestEntityTooLarge)
	})
}

// NewGRPC forwards inference RPCs over cleartext HTTP/2.
func NewGRPC(addr string, maxMessageBytes int64) http.Handler {
	protocols := new(http.Protocols)
	protocols.SetUnencryptedHTTP2(true)
	// Each inference request carries one length-prefixed message.
	return newProxy(addr, maxMessageBytes+grpcframe.HeaderLen, protocols, func(w http.ResponseWriter) {
		grpcframe.WriteError(w, codes.ResourceExhausted, "request body too large")
	})
}

func newProxy(addr string, maxBytes int64, protocols *http.Protocols, rejectTooLarge func(http.ResponseWriter)) http.Handler {
	target := &url.URL{Scheme: "http", Host: addr}
	// The default flush policy streams SSE and preserves trailers-only gRPC errors.
	proxy := httputil.ReverseProxy{
		Transport: &http.Transport{
			Protocols:          protocols,
			DisableCompression: true,
			IdleConnTimeout:    90 * time.Second,
		},
		Rewrite: func(r *httputil.ProxyRequest) {
			r.SetURL(target)
			r.Out.URL.RawQuery = r.In.URL.RawQuery
		},
		ErrorHandler: func(w http.ResponseWriter, r *http.Request, err error) {
			if _, ok := errors.AsType[*http.MaxBytesError](err); ok {
				rejectTooLarge(w)
				return
			}
			if r.Context().Err() == nil {
				log.FromContext(r.Context()).Error(err, "frontend request failed")
			}
			w.WriteHeader(http.StatusBadGateway)
		},
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.ContentLength > maxBytes {
			rejectTooLarge(w)
			return
		}
		r.Body = http.MaxBytesReader(w, r.Body, maxBytes)
		proxy.ServeHTTP(w, r) //#nosec G704 -- the frontend origin is configured at startup
	})
}
