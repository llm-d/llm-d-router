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

package main

import (
	"context"
	"errors"
	"fmt"
	"net"
	"net/http"

	"golang.org/x/sync/errgroup"

	"github.com/llm-d/llm-d-router/internal/frontendproxy"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/server"
)

func withFrontendIngress(cfg config.ServerConfig) server.Option {
	maxBodyMB := cfg.MaxRequestBodySize
	if maxBodyMB == 0 {
		maxBodyMB = config.DefaultMaxRequestBodySize
	}
	maxBytes := maxBodyMB * config.BytesPerMB
	httpProxy := frontendproxy.NewHTTP("127.0.0.1:8000", maxBytes)
	grpcProxy := frontendproxy.NewGRPC("127.0.0.1:50051", maxBytes)
	postRoutes := map[string]http.Handler{
		reqcommon.PathChatCompletions: httpProxy,
		reqcommon.PathCompletions:     httpProxy,
		reqcommon.PathResponses:       httpProxy,

		vllmpb.Inference_Generate_FullMethodName:       grpcProxy,
		vllmpb.Inference_GenerateStream_FullMethodName: grpcProxy,
	}
	return server.WithIngress(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if proxy := postRoutes[r.URL.Path]; r.Method == http.MethodPost && proxy != nil {
				proxy.ServeHTTP(w, r)
				return
			}
			next.ServeHTTP(w, r)
		})
	})
}

func runWithFrontend(ctx context.Context, srv *server.Server, cfg config.ServerConfig, publicListener, callbackListener net.Listener) error {
	if callbackListener == nil {
		var err error
		callbackListener, err = net.Listen("tcp", "127.0.0.1:8081")
		if err != nil {
			return fmt.Errorf("generation listener: %w", err)
		}
	}
	protocols := new(http.Protocols)
	protocols.SetUnencryptedHTTP2(true)
	callback := &http.Server{
		Handler:     srv.VLLMGenerationHandler(),
		Protocols:   protocols,
		ReadTimeout: cfg.ReadTimeout,
	}

	group, groupCtx := errgroup.WithContext(ctx)
	group.Go(func() error {
		if err := callback.Serve(callbackListener); !errors.Is(err, http.ErrServerClosed) {
			return fmt.Errorf("generation listener: %w", err)
		}
		return nil
	})
	group.Go(func() error {
		// Keep the callback available until public requests finish draining.
		err := run(groupCtx, srv, cfg, publicListener)
		return errors.Join(err, callback.Close())
	})
	return group.Wait()
}
