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
	// droppedClientHeaders are client headers the step does not forward: a
	// client Prefer: reserve-endpoint would make EPP answer the pinned prefill
	// request instead of running it, and the connector names the headers that
	// conflict with its transfer fields.
	droppedClientHeaders []string
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
		prefill:              &PrefillStep{useOpenAIFormat: useOpenAI, gwClient: gwClient, kv: concurrent, ec: ecConn},
		decode:               &DecodeStep{gwClient: gwClient, kv: concurrent},
		kv:                   concurrent,
		droppedClientHeaders: append([]string{routing.PreferHeader}, concurrent.ConflictingClientHeaders()...),
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

	if err := s.kv.ApplyTransferFields(ctx, prefillHostPort, prefillBody, reqCtx.Body); err != nil {
		return fmt.Errorf("prefill-decode: %w", err)
	}
	// Marshal prefill before prepareDecodeBody: the two bodies share nested
	// values, which prepareDecodeBody mutates in place.
	prefillBytes, err := json.Marshal(prefillBody)
	if err != nil {
		return fmt.Errorf("prefill-decode: marshal: %w", err)
	}
	if err := s.decode.prepareDecodeBody(ctx, reqCtx); err != nil {
		return fmt.Errorf("prefill-decode: %w", err)
	}

	sendCtx, cancel := context.WithCancelCause(ctx)
	defer cancel(nil)
	decodeReq, err := newDecodeProxyRequest(sendCtx, logger, PrefillDecodeStepName, reqCtx, s.decode.gwClient, reqCtx.Body,
		s.gatewayHeaders(reqCtx, gateway.PhaseDecode, ""))
	if err != nil {
		return err
	}

	logger.V(logutil.DEFAULT).Info("sending requests", "path", path, "prefillEndpoint", prefillHostPort, "stream", reqCtx.Stream)
	prefillDone := make(chan error, 1)
	go func() {
		err := s.sendPrefill(sendCtx, logger, reqCtx, path, prefillBytes, prefillHostPort)
		if err != nil {
			cancel(errPeerRequestFailed)
		}
		prefillDone <- err
	}()

	out, aborted := proxyDecode(logger, s.decode.gwClient.Transport(), reqCtx.ResponseWriter, decodeReq, coordmetrics.UpstreamDecode, nil)
	decodeErr := out.streamedError(PrefillDecodeStepName)
	if decodeErr != nil || aborted {
		cancel(errPeerRequestFailed)
	}
	prefillErr := <-prefillDone

	// The client got a part of the decode response. Under an HTTP server the
	// proxy has aborted the client connection; outside one it returns with the
	// copy stopped. Both end in the abort, so the client sees the truncation.
	if aborted || (decodeErr == nil && out.Status != 0 && !out.BodyComplete) {
		if prefillErr != nil && ctx.Err() == nil {
			logger.Error(prefillErr, "prefill failed after decode started streaming")
		}
		panic(http.ErrAbortHandler)
	}

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
	default:
		logger.Error(prefillErr, "prefill failed after decode completed")
		return nil
	}
}

// gatewayHeaders returns the headers of a request to the phase profile: the
// forwarded client headers without the dropped ones, and the client's request
// id plus idSuffix.
func (s *PrefillDecodeStep) gatewayHeaders(reqCtx *pipeline.RequestContext, phase, idSuffix string) map[string]string {
	headers := gatewayHeaders(reqCtx, phase)
	for _, name := range s.droppedClientHeaders {
		delete(headers, name)
	}
	headers[reqcommon.RequestIDHeaderKey] += idSuffix
	return headers
}

// reserve asks EPP which prefill endpoint it would pick for body, without
// sending the request there, and returns that endpoint's <ip:port>. EPP
// answers the ask with 204 and the endpoint on a response header.
func (s *PrefillDecodeStep) reserve(ctx context.Context, logger logr.Logger, reqCtx *pipeline.RequestContext, path string, body []byte) (string, error) {
	logger.V(logutil.DEBUG).Info("reserving prefill endpoint", "path", path)
	resp, err := s.postPrefill(ctx, logger, reqCtx, path, body, coordmetrics.UpstreamReserveEndpoint, reserveRequestIDSuffix,
		routing.PreferHeader, routing.PreferReserveEndpoint, http.StatusNoContent)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	hostPort := resp.Header.Get(routing.ReservedEndpointHeader)
	if hostPort == "" {
		return "", fmt.Errorf("prefill-decode: reserve-endpoint answer has no %s header; is the reserve-endpoint plugin configured in EPP?", routing.ReservedEndpointHeader)
	}
	logger.V(logutil.DEBUG).Info("reserved prefill endpoint", "status", resp.StatusCode, "header", routing.ReservedEndpointHeader, "endpoint", hostPort)
	return hostPort, nil
}

// sendPrefill sends the prefill request pinned to prefillHostPort. Only its
// status is used; decode carries the answer to the client.
func (s *PrefillDecodeStep) sendPrefill(ctx context.Context, logger logr.Logger, reqCtx *pipeline.RequestContext, path string, body []byte, prefillHostPort string) error {
	resp, err := s.postPrefill(ctx, logger, reqCtx, path, body, coordmetrics.UpstreamPrefill, prefillRequestIDSuffix,
		routing.EndpointPinHeader, prefillHostPort, http.StatusOK)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	// Drain so the connection can be reused.
	_, _ = io.Copy(io.Discard, resp.Body)
	logger.V(logutil.DEBUG).Info("prefill complete")
	return nil
}

// postPrefill sends body to the prefill profile with header set to value. The
// request id is the client's plus idSuffix, and upstream labels the call's
// metrics and names the request in its log line. A status other than
// wantStatus is an error; on success the caller closes the body.
func (s *PrefillDecodeStep) postPrefill(ctx context.Context, logger logr.Logger, reqCtx *pipeline.RequestContext, path string, body []byte,
	upstream, idSuffix, header, value string, wantStatus int) (*http.Response, error) {
	headers := s.gatewayHeaders(reqCtx, gateway.PhasePrefill, idSuffix)
	headers[header] = value
	return postToGateway(ctx, logger, s.prefill.gwClient, gatewayRequest{
		logMsg:     upstream + " request",
		step:       PrefillDecodeStepName,
		upstream:   upstream,
		path:       path,
		body:       body,
		headers:    headers,
		wantStatus: wantStatus,
	})
}
