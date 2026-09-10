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

package steps

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"net/http"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"

	"github.com/llm-d/llm-d-router/pkg/coordinator/engine"
	"github.com/llm-d/llm-d-router/pkg/coordinator/engine/vllm"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const RenderStepName = "render"

func init() {
	pipeline.Register(RenderStepName, NewRenderStep)
}

type RenderStep struct {
	serviceAddress string
	client         *http.Client
	engine         vllm.Engine
}

func NewRenderStep(_ *gateway.Client, params map[string]any) (pipeline.Step, error) {
	timeout := 30 * time.Second
	if v, ok, err := paramDuration(params, "timeout"); err != nil {
		return nil, err
	} else if ok {
		timeout = v
	}

	maxIdleConnsPerHost := 100
	if v, ok, err := paramInt(params, "max_idle_conns_per_host"); err != nil {
		return nil, err
	} else if ok {
		if v < 0 {
			return nil, fmt.Errorf("max_idle_conns_per_host must be non-negative, got %d", v)
		}
		maxIdleConnsPerHost = v
	}

	idleConnTimeout := 90 * time.Second
	if v, ok, err := paramDuration(params, "idle_conn_timeout"); err != nil {
		return nil, err
	} else if ok {
		idleConnTimeout = v
	}

	address, err := paramString(params, "address")
	if err != nil {
		return nil, err
	}

	maxTokens := 0
	if v, ok, err := paramInt(params, "max_total_tokens"); err != nil {
		return nil, err
	} else if ok {
		if v < 0 {
			return nil, fmt.Errorf("max_total_tokens must be non-negative, got %d", v)
		}
		maxTokens = v
	}

	maxPlaceholders := 0
	if v, ok, err := paramInt(params, "max_total_placeholder_tokens"); err != nil {
		return nil, err
	} else if ok {
		if v < 0 {
			return nil, fmt.Errorf("max_total_placeholder_tokens must be non-negative, got %d", v)
		}
		maxPlaceholders = v
	}

	selectedEngine := vllm.New(false, engine.Limits{MaxTotalTokens: maxTokens, MaxTotalPlaceholderTokens: maxPlaceholders})

	transport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConnsPerHost: maxIdleConnsPerHost,
		IdleConnTimeout:     idleConnTimeout,
		ForceAttemptHTTP2:   true,
	}

	return &RenderStep{
		serviceAddress: address,
		engine:         selectedEngine,
		client:         &http.Client{Timeout: timeout, Transport: transport},
	}, nil
}

func (s *RenderStep) SetServiceAddress(addr string) {
	s.serviceAddress = addr
}

func (s *RenderStep) Name() string { return RenderStepName }

func (s *RenderStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	prepared, err := s.engine.PrepareRender(ctx, reqCtx)
	if err != nil {
		return err
	}
	if prepared != nil {
		if err := s.postRender(ctx, reqCtx, prepared); err != nil {
			return err
		}
	}
	return nil
}

// postRender sends the engine request and applies its renderer response.
func (s *RenderStep) postRender(ctx context.Context, reqCtx *pipeline.RequestContext, prepared *engine.Request) error {
	if s.serviceAddress == "" {
		return errors.New("render: service address not configured (set 'address' in render step params)")
	}

	logger := log.FromContext(ctx).WithName(RenderStepName)

	body, err := json.Marshal(prepared.Body)
	if err != nil {
		return fmt.Errorf("marshaling request for render: %w", err)
	}
	url := s.serviceAddress + prepared.Path
	logger.V(logutil.DEFAULT).Info("sending request", "url", url)

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("creating render request: %w", err)
	}
	req.ContentLength = int64(len(body))
	req.Header.Set(gateway.ContentTypeHeader, gateway.ContentTypeJSON)
	for k, v := range reqCtx.ForwardedHeaders() {
		req.Header.Set(k, v)
	}

	call := coordmetrics.StartUpstreamCall(coordmetrics.UpstreamRender)
	resp, err := s.client.Do(req)
	call.Done()
	if err != nil {
		return fmt.Errorf("render request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		respBody := readErrorBody(resp.Body)
		return upstreamError(RenderStepName, resp.StatusCode, respBody)
	}
	return s.engine.ApplyRenderResponse(ctx, reqCtx, resp.Body)
}
