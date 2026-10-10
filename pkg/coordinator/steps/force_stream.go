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
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"maps"
	"math"
	"net/http"
	"strconv"

	"github.com/dustin/go-humanize"
	"github.com/go-logr/logr"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/common/httplog"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// defaultForceStreamBufferSize is the budget applied when force_stream is on but
// force_stream_buffer_size is unset. 1GiB of buffered responses is negligible on
// a node that runs GPU inference, so an operator overrides it only to tune.
const defaultForceStreamBufferSize = "1GiB"

// defaultForceStreamMaxRequestBytes caps one forced request's reservation when
// force_stream_max_request_size is unset. 256KiB holds roughly 16k output tokens
// for a single choice, enough for a large reply while keeping one request from
// claiming a large share of the shared budget. It is a fixed size rather than a
// fraction of the budget: the cap bounds a single reply, not a slice of the
// aggregate. Capped to the budget when that is smaller.
const defaultForceStreamMaxRequestBytes int64 = 256 << 10

const (
	// forceStreamBytesPerToken bounds the decoded UTF-8 bytes one output token
	// contributes. A token's text spans a few code points and UTF-8 uses at
	// most four bytes per code point, so this is an upper bound with margin, not
	// an average. The per-request reservation and ceiling derive from it, so a
	// response that honors its token limit is never aborted; an upstream that
	// ignores the limit and streams without bound still trips the ceiling.
	forceStreamBytesPerToken = 16
	// forceStreamPerChoiceOverheadBytes covers one choice's JSON envelope (its
	// index, finish_reason, and message or text keys) so a request asking for
	// many short completions still reserves enough for their structure.
	forceStreamPerChoiceOverheadBytes = 256
	// forceStreamBaseOverheadBytes covers the top-level envelope shared by every
	// reply: id, model, created, object, and the usage block.
	forceStreamBaseOverheadBytes = 2 << 10
	// forceStreamMaxFrameBytes caps one SSE line. Streaming chunks are small
	// (one token delta), so this bounds a single malformed or oversized frame
	// rather than a whole response.
	forceStreamMaxFrameBytes = 1 << 20
	// forceStreamScanStartBytes is the SSE scanner's initial buffer size; it
	// grows as needed toward the per-request frame cap.
	forceStreamScanStartBytes = 64 << 10
)

// errForceStreamCeiling marks a forced request aborted because its buffered
// response exceeded the bytes reserved for it. The server answers a clean 5xx:
// nothing was written to the client when this is returned.
var errForceStreamCeiling = errors.New("force-stream: buffered response exceeded its reserved budget")

// sseDataPrefix and sseDoneMarker delimit the SSE frames the upstream streams.
// Each data line carries one JSON object; the stream ends with data: [DONE].
// sseDataPrefix is intentionally the bare "data:" without the trailing space
// reqcommon.SSEDataPrefix carries, since ssePayload trims surrounding space and
// so matches both framings.
var (
	sseDataPrefix = []byte("data:")
	sseDoneMarker = []byte(reqcommon.SSEDoneMarker)
)

// parseForceStreamBudget reads force_stream_buffer_size into a byte budget,
// applying defaultForceStreamBufferSize when the operator leaves it unset. The
// value is a human-readable size ("512MiB", "1GiB"); a string that fails to
// parse, is zero, or exceeds int64 is a configuration error rather than a silent
// fallback, so a mistyped budget fails config load instead of running a default.
func parseForceStreamBudget(params map[string]any) (*forceStreamBudget, error) {
	sizeStr, err := paramString(params, ParamForceStreamBufferSize)
	if err != nil {
		return nil, err
	}
	if sizeStr == "" {
		sizeStr = defaultForceStreamBufferSize
	}
	size, err := humanize.ParseBytes(sizeStr)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", ParamForceStreamBufferSize, err)
	}
	if size == 0 {
		return nil, fmt.Errorf("%s: must be greater than zero", ParamForceStreamBufferSize)
	}
	if size > math.MaxInt64 {
		return nil, fmt.Errorf("%s: %q exceeds the maximum budget", ParamForceStreamBufferSize, sizeStr)
	}
	budgetMax := int64(size)

	perRequest, err := parseForceStreamPerRequest(params, budgetMax)
	if err != nil {
		return nil, err
	}
	return newForceStreamBudget(budgetMax, perRequest), nil
}

// parseForceStreamPerRequest reads force_stream_max_request_size into the
// per-request reservation cap. Unset defaults to defaultForceStreamMaxRequestBytes
// (capped to the shared budget when that is smaller), so one request cannot claim
// a large share of the budget without the operator opting in. An explicit value
// must be in (0, max]: a cap above the shared budget could never bind, so it fails
// config load rather than clamping silently.
func parseForceStreamPerRequest(params map[string]any, budgetMax int64) (int64, error) {
	sizeStr, err := paramString(params, ParamForceStreamMaxRequestSize)
	if err != nil {
		return 0, err
	}
	if sizeStr == "" {
		if budgetMax < defaultForceStreamMaxRequestBytes {
			return budgetMax, nil
		}
		return defaultForceStreamMaxRequestBytes, nil
	}
	size, err := humanize.ParseBytes(sizeStr)
	if err != nil {
		return 0, fmt.Errorf("%s: %w", ParamForceStreamMaxRequestSize, err)
	}
	if size == 0 {
		return 0, fmt.Errorf("%s: must be greater than zero", ParamForceStreamMaxRequestSize)
	}
	if size > math.MaxInt64 {
		return 0, fmt.Errorf("%s: %q exceeds the maximum budget", ParamForceStreamMaxRequestSize, sizeStr)
	}
	perRequest := int64(size)
	if perRequest > budgetMax {
		return 0, fmt.Errorf("%s: %q exceeds %s", ParamForceStreamMaxRequestSize, sizeStr, ParamForceStreamBufferSize)
	}
	return perRequest, nil
}

// estimateReservation sizes the buffer one forced request may hold and reports
// whether force-streaming it is safe. ok is false when the body names no output
// token limit (an unbounded response cannot be reserved) or when its estimate
// exceeds the per-request cap (one request must never claim more than its share
// of the shared budget). The estimate reserves n completions, each tokenLimit
// tokens, plus the reply envelope; over-reserving only sends more requests to the
// pass-through, never past the budget.
func (s *DecodeStep) estimateReservation(reqCtx *pipeline.RequestContext) (int64, bool) {
	apiType := reqcommon.DetectAPIType(reqCtx.OriginalPath)
	tokenLimit, ok := reqcommon.OutputTokenLimit(reqCtx.Body, apiType)
	if !ok {
		return 0, false
	}
	n := reqcommon.OutputChoiceCount(reqCtx.Body, apiType)

	maxBytes := s.budget.perRequestMax
	if maxBytes <= forceStreamBaseOverheadBytes {
		return 0, false
	}
	contentBudget := maxBytes - forceStreamBaseOverheadBytes

	// Each guard below both enforces the budget ceiling and keeps the following
	// multiplication inside int64: a client-supplied tokenLimit or n large
	// enough to overflow first exceeds contentBudget here and returns false.
	if int64(tokenLimit) > (contentBudget-forceStreamPerChoiceOverheadBytes)/forceStreamBytesPerToken {
		return 0, false
	}
	perChoice := int64(tokenLimit)*forceStreamBytesPerToken + forceStreamPerChoiceOverheadBytes
	if int64(n) > contentBudget/perChoice {
		return 0, false
	}
	return int64(n)*perChoice + forceStreamBaseOverheadBytes, true
}

// executeForceStream sends the prepared decode body upstream with streaming
// enabled, reassembles the streamed frames into one non-streaming reply, and
// writes that reply to the client. It balances the reservation on every exit.
//
// Nothing reaches the client until the whole response is buffered, so an error
// before the final write leaves the client uncommitted: a transport failure or
// a ceiling abort returns a plain error and the server answers a clean 5xx. An
// upstream status in the 4xx/5xx range is forwarded verbatim and reported with
// UpstreamStreamedError so the server does not overwrite it.
func (s *DecodeStep) executeForceStream(ctx context.Context, logger logr.Logger, reqCtx *pipeline.RequestContext, shape sseShape, reserved int64) error {
	defer s.budget.release(reserved)

	bodyBytes, headers, err := buildForceStreamRequest(reqCtx, shape)
	if err != nil {
		return err
	}

	logger.V(logutil.DEFAULT).Info("force-streaming request", "path", reqCtx.OriginalPath, "reservedBytes", reserved)
	if v := logger.V(logutil.DEBUG); v.Enabled() {
		v.Info("force-stream request body", "method", "POST", "path", reqCtx.OriginalPath, "bodyLen", len(bodyBytes), "headers", httplog.RedactedHeaders(headers))
	}

	call := coordmetrics.StartUpstreamCall(coordmetrics.UpstreamDecode)
	resp, err := s.gwClient.Post(ctx, reqCtx.OriginalPath, bodyBytes, headers)
	call.Done()
	if err != nil {
		// No response headers arrived, so nothing is on the wire; a plain error
		// lets the server answer a clean 502.
		return fmt.Errorf("%s: force-stream request: %w", DecodeStepName, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		return forwardUpstreamError(reqCtx, resp)
	}

	n := reqcommon.OutputChoiceCount(reqCtx.Body, reqcommon.DetectAPIType(reqCtx.OriginalPath))
	reassembler := newSSEReassembler(shape, n)
	if err := scanForcedResponse(resp.Body, reassembler, reserved); err != nil {
		if errors.Is(err, errForceStreamCeiling) {
			coordmetrics.IncForceStreamTotal(reqCtx.Model, coordmetrics.ForceStreamResultErrorCeiling)
			logger.Error(err, "force-stream aborted", "reservedBytes", reserved)
		}
		return fmt.Errorf("%s: %w", DecodeStepName, err)
	}

	return writeForcedResponse(logger, reqCtx, reassembler)
}

// forceStreamLossy reports whether force-streaming would drop fields the
// reassembler does not fold. The chat and text shapes fold only generated text,
// so a request that may produce tool calls, function calls, or logprobs takes
// the pass-through to keep them. The responses shape emits the upstream object
// verbatim (keeping those fields) and generate has none, so neither is lossy.
func forceStreamLossy(shape sseShape, body map[string]any) bool {
	if shape != sseShapeChat && shape != sseShapeText {
		return false
	}
	if tools, ok := body[reqcommon.FieldTools].([]any); ok && len(tools) > 0 {
		return true
	}
	if fns, ok := body[reqcommon.FieldFunctions].([]any); ok && len(fns) > 0 {
		return true
	}
	switch lp := body[reqcommon.FieldLogprobs].(type) {
	case bool:
		return lp
	case float64, int:
		// On the legacy Completions API logprobs is a count, and the field's
		// presence (including the value 0) requests the sampled token's logprob.
		return true
	}
	return false
}

// buildForceStreamRequest marshals the forced (stream:true) decode body and
// builds its forwarding headers, shared by the buffered and incremental paths.
func buildForceStreamRequest(reqCtx *pipeline.RequestContext, shape sseShape) ([]byte, map[string]string, error) {
	bodyBytes, err := json.Marshal(forceStreamBody(reqCtx.Body, shape))
	if err != nil {
		return nil, nil, fmt.Errorf("%s: force-stream marshal: %w", DecodeStepName, err)
	}
	headers := gatewayHeaders(reqCtx, gateway.PhaseDecode)
	return bodyBytes, headers, nil
}

// forceStreamBody clones the prepared decode body, enables streaming, and asks
// the upstream to emit a trailing usage frame. The clone keeps the shared
// reqCtx.Body (and any nested stream_options the client sent) unmutated.
func forceStreamBody(src map[string]any, shape sseShape) map[string]any {
	body := maps.Clone(src)
	body[reqcommon.FieldStream] = true

	// Only the chat and text shapes return a usage block, and only when asked, so
	// request one for them. The Responses API has no stream_options (usage arrives
	// in response.completed), and the generate reply carries no usage block.
	if shape == sseShapeChat || shape == sseShapeText {
		opts := map[string]any{}
		if existing, ok := body[reqcommon.FieldStreamOptions].(map[string]any); ok {
			maps.Copy(opts, existing)
		}
		opts["include_usage"] = true
		body[reqcommon.FieldStreamOptions] = opts
	}
	return body
}

// maxFrameBytes bounds a single SSE line, holding the scanner's transient buffer
// within the request's reservation (the bytes the budget accounts for). The
// per-token chunk shapes also keep the small-frame guard: their frames are small,
// so a frame near the reservation or 1 MiB is malformed. The Responses shape
// delivers its whole reply in one response.completed frame, so its only bound is
// the reservation.
func maxFrameBytes(shape sseShape, reserved int64) int {
	if shape != sseShapeResponses && reserved > forceStreamMaxFrameBytes {
		return forceStreamMaxFrameBytes
	}
	return int(reserved)
}

// initialScanBuf sizes the scanner's starting buffer so the effective maximum
// frame size is frameCap. bufio caps a token at the larger of its max argument
// and the starting buffer's capacity, so a starting capacity above frameCap would
// let the buffer grow past the reservation.
func initialScanBuf(frameCap int) int {
	if frameCap < forceStreamScanStartBytes {
		return frameCap
	}
	return forceStreamScanStartBytes
}

// scanForcedResponse folds the upstream SSE frames into reassembler, enforcing
// the per-request ceiling. A frame that cannot be parsed or a read failure is an
// upstream fault; exceeding the ceiling returns errForceStreamCeiling. Both
// return before any byte is written to the client.
func scanForcedResponse(r io.Reader, reassembler *sseReassembler, reserved int64) error {
	scanner := bufio.NewScanner(r)
	frameCap := maxFrameBytes(reassembler.shape, reserved)
	scanner.Buffer(make([]byte, 0, initialScanBuf(frameCap)), frameCap)
	for scanner.Scan() {
		payload, ok := ssePayload(scanner.Bytes())
		if !ok {
			continue
		}
		if bytes.Equal(payload, sseDoneMarker) {
			break
		}
		var frame map[string]any
		if err := json.Unmarshal(payload, &frame); err != nil {
			return fmt.Errorf("force-stream: parse frame: %w", err)
		}
		reassembler.add(frame)
		if reassembler.bufferedBytes() > reserved {
			return errForceStreamCeiling
		}
	}
	if err := scanner.Err(); err != nil {
		// The scanner's frame cap is the request's reservation (or the 1 MiB
		// guard), so a frame that trips it has overflowed the reserved budget;
		// count it as a ceiling abort rather than a generic read error.
		if errors.Is(err, bufio.ErrTooLong) {
			return errForceStreamCeiling
		}
		return fmt.Errorf("force-stream: read upstream: %w", err)
	}
	return nil
}

// writeForcedResponse marshals the reassembled reply and writes it as one
// non-streaming JSON response. A marshal failure returns a plain error before
// any write, so the server answers a clean 502.
func writeForcedResponse(logger logr.Logger, reqCtx *pipeline.RequestContext, reassembler *sseReassembler) error {
	payload, err := json.Marshal(reassembler.result())
	if err != nil {
		return fmt.Errorf("%s: force-stream marshal response: %w", DecodeStepName, err)
	}

	w := reqCtx.ResponseWriter
	w.Header().Set(gateway.ContentTypeHeader, reqcommon.ContentTypeJSON)
	w.Header().Set("Content-Length", strconv.Itoa(len(payload)))
	w.WriteHeader(http.StatusOK)
	if _, err := w.Write(payload); err != nil {
		// The response is already committed; the client disconnected mid-write.
		// Report success to the pipeline and record the forced outcome, since
		// the request was served as far as the coordinator is concerned.
		logger.Error(err, "force-stream client write incomplete")
	}
	coordmetrics.IncForceStreamTotal(reqCtx.Model, coordmetrics.ForceStreamResultForced)
	logger.V(logutil.DEFAULT).Info("force-stream complete", "bytes", len(payload))
	return nil
}

// forwardUpstreamError relays an upstream 4xx/5xx response to the client with
// its status and body, matching what the non-forced pass-through would stream.
// It returns UpstreamStreamedError so the server records the failure without
// writing another response.
func forwardUpstreamError(reqCtx *pipeline.RequestContext, resp *http.Response) error {
	respBody := readErrorBody(resp.Body)
	contentType := resp.Header.Get(gateway.ContentTypeHeader)
	if contentType == "" {
		contentType = reqcommon.ContentTypeJSON
	}

	w := reqCtx.ResponseWriter
	w.Header().Set(gateway.ContentTypeHeader, contentType)
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(respBody)
	return &pipeline.UpstreamStreamedError{Step: DecodeStepName, StatusCode: resp.StatusCode}
}

// ssePayload extracts the JSON payload of an SSE data line, trimming the data:
// prefix and surrounding whitespace (including the trailing CR a CRLF stream
// leaves). ok is false for any other line (event:, id:, comments, blanks) and
// for a data line with an empty payload.
func ssePayload(line []byte) ([]byte, bool) {
	if !bytes.HasPrefix(line, sseDataPrefix) {
		return nil, false
	}
	payload := bytes.TrimSpace(line[len(sseDataPrefix):])
	if len(payload) == 0 {
		return nil, false
	}
	return payload, true
}
