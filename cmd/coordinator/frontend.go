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
	"net/url"

	"golang.org/x/sync/errgroup"

	"github.com/llm-d/llm-d-router/internal/frontendproxy"
	"github.com/llm-d/llm-d-router/pkg/common/grpc/vllm/gen/vllmpb"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/coordinator/config"
	"github.com/llm-d/llm-d-router/pkg/coordinator/server"
)

func frontendOption(cfg config.ServerConfig) server.Option {
	maxSize := cfg.MaxRequestBodySize
	if maxSize == 0 {
		maxSize = config.DefaultMaxRequestBodySize
	}
	maxBytes := maxSize * config.BytesPerMB
	api := frontendproxy.New(&url.URL{Scheme: "http", Host: "127.0.0.1:8000"}, maxBytes, false)
	rpc := frontendproxy.New(&url.URL{Scheme: "http", Host: "127.0.0.1:50051"}, maxBytes, true)
	return publicFrontend(api, rpc)
}

func publicFrontend(api, rpc http.Handler) server.Option {
	return server.WithIngress(func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method == http.MethodPost {
				switch r.URL.Path {
				case reqcommon.PathChatCompletions, reqcommon.PathCompletions, reqcommon.PathResponses, "/tokenize", "/detokenize":
					api.ServeHTTP(w, r)
					return
				case vllmpb.Inference_Generate_FullMethodName, vllmpb.Inference_GenerateStream_FullMethodName:
					rpc.ServeHTTP(w, r)
					return
				}
			}
			next.ServeHTTP(w, r)
		})
	})
}

func runWithFrontend(ctx context.Context, srv *server.Server, cfg config.ServerConfig, lis net.Listener) error {
	if lis == nil {
		var err error
		lis, err = net.Listen("tcp", "127.0.0.1:8081")
		if err != nil {
			return fmt.Errorf("generation listener: %w", err)
		}
	}
	protocols := new(http.Protocols)
	protocols.SetUnencryptedHTTP2(true)
	callback := &http.Server{Handler: srv.GenerationHandler(), Protocols: protocols, ReadTimeout: cfg.ReadTimeout}
	group, groupCtx := errgroup.WithContext(ctx)
	group.Go(func() error {
		if err := callback.Serve(lis); !errors.Is(err, http.ErrServerClosed) {
			return fmt.Errorf("generation listener: %w", err)
		}
		return nil
	})
	group.Go(func() error {
		err := run(groupCtx, srv, cfg, nil)
		// Public requests may call back while the public listener drains.
		return errors.Join(err, callback.Close())
	})
	return group.Wait()
}
