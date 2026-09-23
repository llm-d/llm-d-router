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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"

	"github.com/llm-d/llm-d-router/pkg/coordinator/connectors/kv"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

const PrefillDecodeStepName = "prefill-decode"

// Suffixes appended to the client request id for the reserve-endpoint and
// prefill requests. EPP plugins key per-request state by x-request-id, so each
// request of one client request needs its own id; decode keeps the client's.
const (
	reserveRequestIDSuffix = "-reserve"
	prefillRequestIDSuffix = "-prefill"
)

// droppedClientHeaders are client headers the step does not forward. A client
// Prefer: reserve-endpoint would make EPP answer the pinned prefill request
// instead of running it, and SGLang runs a request on the rank that
// x-data-parallel-rank names, which can conflict with the rank bootstrap_room
// implies.
var droppedClientHeaders = []string{routing.PreferHeader, "x-data-parallel-rank"}

func init() {
	pipeline.Register(PrefillDecodeStepName, NewPrefillDecodeStep)
}

// PrefillDecodeStep runs prefill and decode for a KV connector whose two
// requests must be in flight together (kv-sglang). It asks EPP which prefill
// endpoint it would pick, writes that endpoint into both bodies, pins the
// prefill request to it, and sends both requests at once. Decode streams to
// the client; when one request fails, the other is cancelled.
type PrefillDecodeStep struct {
	prefill *PrefillStep
	decode  *DecodeStep
	kv      kv.ConcurrentConnector
}

func NewPrefillDecodeStep(gwClient *gateway.Client, params map[string]any) (pipeline.Step, error) {
	if gwClient == nil {
		return nil, errors.New("prefill-decode: gateway client is required")
	}
	useOpenAI, err := parseUseOpenAIFormat(params)
	if err != nil {
		return nil, fmt.Errorf("prefill-decode: %w", err)
	}
	kvConn, err := buildKVConnector(params)
	if err != nil {
		return nil, fmt.Errorf("prefill-decode: %w", err)
	}
	concurrent, ok := kvConn.(kv.ConcurrentConnector)
	if !ok {
		return nil, fmt.Errorf("prefill-decode: kv_connector %q sends prefill and decode one after the other; use the %q and %q steps",
			kvConn.Name(), PrefillStepName, DecodeStepName)
	}
	ecConn, err := buildECConnector(params)
	if err != nil {
		return nil, fmt.Errorf("prefill-decode: %w", err)
	}
	return &PrefillDecodeStep{
		prefill: &PrefillStep{useOpenAIFormat: useOpenAI, gwClient: gwClient, kv: concurrent, ec: ecConn},
		decode:  &DecodeStep{useOpenAIFormat: useOpenAI, gwClient: gwClient, kv: concurrent},
		kv:      concurrent,
	}, nil
}

func (s *PrefillDecodeStep) Name() string { return PrefillDecodeStepName }

func (s *PrefillDecodeStep) Execute(ctx context.Context, reqCtx *pipeline.RequestContext) error {
	logger := log.FromContext(ctx).WithName(PrefillDecodeStepName)

	format := resolveFormat(s.prefill.useOpenAIFormat, reqCtx.OriginalPath)
	path := format.Path()
	prefillBody, err := s.prefill.buildPrefillBody(ctx, reqCtx, format)
	if err != nil {
		return fmt.Errorf("prefill-decode: %w", err)
	}
	reserveBytes, err := json.Marshal(prefillBody)
	if err != nil {
		return fmt.Errorf("prefill-decode: marshal: %w", err)
	}
	prefillHostPort, err := s.reserve(ctx, logger, reqCtx, path, reserveBytes)
	if err != nil {
		return err
	}

	if err := s.kv.ApplyBootstrapFields(ctx, prefillHostPort, prefillBody, reqCtx.Body); err != nil {
		return fmt.Errorf("prefill-decode: %w", err)
	}
	// Marshal prefill before prepareDecodeBody: the two bodies share nested
	// values, which prepareDecodeBody mutates in place.
	prefillBytes, err := json.Marshal(prefillBody)
	if err != nil {
		return fmt.Errorf("prefill-decode: marshal: %w", err)
	}
	s.decode.prepareDecodeBody(ctx, reqCtx)

	legCtx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	decodeReq, err := newDecodeProxyRequest(legCtx, logger, PrefillDecodeStepName, reqCtx, s.decode.gwClient, reqCtx.Body, nil)
	if err != nil {
		return err
	}
	for _, name := range droppedClientHeaders {
		decodeReq.Header.Del(name)
	}

	logger.V(logutil.DEFAULT).Info("sending requests", "path", path, "prefillEndpoint", prefillHostPort, "stream", reqCtx.Stream)
	prefillDone := make(chan error, 1)
	go func() {
		err := s.sendPrefill(legCtx, logger, reqCtx, path, prefillBytes, prefillHostPort)
		if err != nil {
			cancel(errPeerLegFailed)
		}
		prefillDone <- err
	}()

	out := serveDecode(logger, s.decode.gwClient, reqCtx, decodeReq)
	decodeErr := out.streamedError(PrefillDecodeStepName)
	if decodeErr != nil {
		cancel(errPeerLegFailed)
	}
	prefillErr := <-prefillDone

	switch {
	case decodeErr != nil:
		return decodeErr
	case prefillErr == nil:
		logger.V(logutil.DEFAULT).Info("complete")
		return nil
	case ctx.Err() != nil:
		// The client went away and both requests were cancelled with it. The
		// decode proxy has already handled that, as in the decode step.
		return nil
	case out.Status == 0:
		// Decode wrote nothing, so the client gets the prefill error.
		return prefillErr
	case out.BodyComplete:
		logger.Error(prefillErr, "prefill failed after decode completed")
		return nil
	default:
		// Decode had started to answer; the client got a truncated response.
		streamed := &pipeline.UpstreamStreamedError{Step: PrefillDecodeStepName, Cause: prefillErr}
		var upstream *pipeline.UpstreamError
		if errors.As(prefillErr, &upstream) {
			streamed.StatusCode = upstream.StatusCode
		}
		return streamed
	}
}

// reserve asks EPP which prefill endpoint it would pick for body, without
// sending the request there, and returns that endpoint's <ip:port>.
func (s *PrefillDecodeStep) reserve(ctx context.Context, logger logr.Logger, reqCtx *pipeline.RequestContext, path string, body []byte) (string, error) {
	resp, err := s.postPrefill(ctx, reqCtx, path, body, coordmetrics.UpstreamReserveEndpoint, reserveRequestIDSuffix,
		routing.PreferHeader, routing.PreferReserveEndpoint)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	hostPort := resp.Header.Get(routing.ReservedEndpointHeader)
	if hostPort == "" {
		return "", fmt.Errorf("prefill-decode: reserve-endpoint answer has no %s header; is the reserve-endpoint plugin configured in EPP?", routing.ReservedEndpointHeader)
	}
	logger.V(logutil.DEBUG).Info("reserved prefill endpoint", "endpoint", hostPort)
	return hostPort, nil
}

// sendPrefill sends the prefill request pinned to prefillHostPort. Only its
// status is used; decode carries the answer to the client.
func (s *PrefillDecodeStep) sendPrefill(ctx context.Context, logger logr.Logger, reqCtx *pipeline.RequestContext, path string, body []byte, prefillHostPort string) error {
	resp, err := s.postPrefill(ctx, reqCtx, path, body, coordmetrics.UpstreamPrefill, prefillRequestIDSuffix,
		routing.PrefillPinHeader, prefillHostPort)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	// Drain so the connection can be reused.
	_, _ = io.Copy(io.Discard, resp.Body)
	logger.V(logutil.DEBUG).Info("prefill complete")
	return nil
}

// postPrefill sends body to the prefill profile with the forwarded client
// headers, without droppedClientHeaders, plus header set to value. The request
// id is the client's plus idSuffix, and upstream labels the call's metrics. A
// status other than 200 is an error; on success the caller closes the body.
func (s *PrefillDecodeStep) postPrefill(ctx context.Context, reqCtx *pipeline.RequestContext, path string, body []byte,
	upstream, idSuffix, header, value string) (*http.Response, error) {
	headers := reqCtx.ForwardedHeaders()
	for _, name := range droppedClientHeaders {
		delete(headers, name)
	}
	headers[reqcommon.RequestIDHeaderKey] = reqCtx.RequestID + idSuffix
	headers[gateway.EPPProfileHeader] = gateway.PhasePrefill
	headers[header] = value

	call := coordmetrics.StartUpstreamCall(upstream)
	resp, err := s.prefill.gwClient.Post(ctx, path, body, headers)
	call.Done()
	if err != nil {
		return nil, fmt.Errorf("prefill-decode: %s request: %w", upstream, err)
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		return nil, upstreamError(PrefillDecodeStepName, resp.StatusCode, readErrorBody(resp.Body))
	}
	return resp, nil
}
