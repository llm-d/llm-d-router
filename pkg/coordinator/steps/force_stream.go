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
	// forceStreamBytesPerToken is the per-token byte unit the reservation and the
	// ceiling both derive from, so a response that honors its token limit is never
	// aborted and an upstream that ignores it still trips the ceiling. It bounds the
	// two encodings a token can take. As text it spans a few code points at up to
	// four UTF-8 bytes each, so 16 is an upper bound with margin. As a generate-shape
	// token id it bounds the id's JSON encoding (decimal digits and a separator),
	// which stays under 16 bytes for any realistic vocabulary. Because the
	// reservation uses this same unit, the generate ceiling trips at the token limit
	// regardless of the exact per-id width.
	forceStreamBytesPerToken = 16
	// forceStreamPerChoiceOverheadBytes covers one choice's JSON envelope (its
	// index, finish_reason, and message or text keys) so a request asking for
	// many short completions still reserves enough for their structure.
	forceStreamPerChoiceOverheadBytes = 256
	// forceStreamBaseOverheadBytes covers the top-level envelope shared by every
	// reply: id, model, created, object, and the usage block.
	forceStreamBaseOverheadBytes = 2 << 10
	// forceStreamMaxFrameBytes caps one SSE line on the incremental scanner and is
	// the upper bound for the buffered chunk shapes, whose per-request reservation is
	// usually smaller and binds first. Streaming chunks are small (one token delta),
	// so this rejects a single malformed or oversized frame rather than a whole reply.
	forceStreamMaxFrameBytes = 1 << 20
	// forceStreamScanStartBytes is the SSE scanner's initial buffer size; it
	// grows as needed toward the per-request frame cap.
	forceStreamScanStartBytes = 64 << 10
)

// errForceStreamCeiling marks a forced request aborted because its buffered
// response exceeded the bytes reserved for it. The server answers a clean 5xx:
// nothing was written to the client when this is returned.
var errForceStreamCeiling = errors.New("force-stream: buffered response exceeded its reserved budget")

// errForceStreamOversize marks a Responses forced reply whose terminal frame did
// not fit the reservation. The reservation prices generated tokens, not the
// request-derived fields (tools, instructions, schema, metadata) a terminal frame
// echoes, so the shortfall is a coordinator estimate miss, not a pod fault.
// Nothing has reached the client, so the decode step falls back to the
// pass-through rather than failing a reply the pod returned intact.
var errForceStreamOversize = errors.New("force-stream: responses reply exceeded its reserved budget; falling back to pass-through")

// errForceStreamUnreadable marks a forced request whose upstream returned 200 but
// a stream the coordinator could not parse or read before any bytes reached the
// client. The non-forced pass-through may serve it from a different pod code path,
// so the decode step falls back to it rather than failing. The underlying parse or
// read error is logged where this is returned, so the fallback stays diagnosable.
var errForceStreamUnreadable = errors.New("force-stream: upstream stream unreadable; falling back to pass-through")

// errForceStreamTruncated marks a forced stream that ended with a clean EOF before
// any terminal marker: no data: [DONE] line and no shape terminal (a Responses
// terminal event or a choice finish_reason). The reassembly would look complete
// but omit the cut-off tail. The buffered path falls back to the pass-through; the
// incremental path falls back while still uncommitted, or truncates the committed
// connection rather than writing a clean suffix. It is handled like
// errForceStreamUnreadable.
var errForceStreamTruncated = errors.New("force-stream: upstream stream ended before a terminal marker")

// sseDataPrefix and sseDoneMarker delimit the SSE frames the upstream streams.
// Each data line carries one JSON object; the stream ends with data: [DONE].
// sseDataPrefix drops the trailing space reqcommon.SSEDataPrefix carries so it
// matches both "data:" and "data: " framings (ssePayload trims the remainder); it
// is derived from the shared constant so the two cannot diverge.
var (
	sseDataPrefix = bytes.TrimRight([]byte(reqcommon.SSEDataPrefix), " ")
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
	if budgetMax <= forceStreamBaseOverheadBytes {
		return nil, fmt.Errorf("%s: %q does not exceed the %d-byte reply-envelope overhead, so no request could buffer output", ParamForceStreamBufferSize, sizeStr, forceStreamBaseOverheadBytes)
	}

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
// must leave room for output above the reply envelope and must not exceed the
// shared budget: a cap at or below the envelope overhead would force-stream
// nothing, and one above the budget could never bind, so either fails config load
// rather than silently degrading.
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
	if perRequest <= forceStreamBaseOverheadBytes {
		return 0, fmt.Errorf("%s: %q does not exceed the %d-byte reply-envelope overhead, so no output could be buffered", ParamForceStreamMaxRequestSize, sizeStr, forceStreamBaseOverheadBytes)
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
		coordmetrics.IncForceStreamTotal(reqCtx.Model, coordmetrics.ForceStreamResultErrorTransport)
		return fmt.Errorf("%s: force-stream request: %w", DecodeStepName, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		coordmetrics.IncForceStreamTotal(reqCtx.Model, coordmetrics.ForceStreamResultErrorUpstream)
		return forwardUpstreamError(reqCtx, resp)
	}

	n := reqcommon.OutputChoiceCount(reqCtx.Body, reqcommon.DetectAPIType(reqCtx.OriginalPath))
	reassembler := newSSEReassembler(shape, n)
	if err := scanForcedResponse(resp.Body, reassembler, reserved); err != nil {
		switch {
		case errors.Is(err, errForceStreamCeiling):
			coordmetrics.IncForceStreamTotal(reqCtx.Model, coordmetrics.ForceStreamResultErrorCeiling)
			logger.Error(err, "force-stream aborted", "reservedBytes", reserved)
		case errors.Is(err, errForceStreamOversize):
			// decode records fallback_oversize and runs the pass-through.
		default:
			// The upstream returned 200 but no complete reply reached the client: an
			// unreadable stream (bad JSON, interrupted read) or one cut off before a
			// terminal marker. Log the detail; decode records fallback_unreadable and
			// runs the pass-through.
			logger.Error(err, "force-stream fell back: incomplete upstream stream")
			return errForceStreamUnreadable
		}
		return fmt.Errorf("%s: %w", DecodeStepName, err)
	}

	return writeForcedResponse(logger, reqCtx, reassembler)
}

// forceStreamLossy reports whether force-streaming would drop fields the
// reassembler does not fold. The chat and text shapes fold only generated text,
// so a request that may produce tool calls, function calls, logprobs, or chat
// reasoning content takes the pass-through to keep them. The responses shape
// emits the upstream object verbatim (keeping those fields) and generate has
// none, so neither is lossy.
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
	// vLLM emits chat reasoning as delta.reasoning_content, folded into
	// message.reasoning_content, which the reassembler drops. The mode is
	// activated by chat_template_kwargs.enable_thinking or by reasoning_effort, so
	// a chat request carrying either keeps its reasoning via the pass-through.
	if shape == sseShapeChat {
		if kwargs, ok := body[reqcommon.FieldChatTemplateKwargs].(map[string]any); ok {
			if thinking, ok := kwargs[reqcommon.FieldEnableThinking].(bool); ok && thinking {
				return true
			}
		}
		if effort, ok := body[reqcommon.FieldReasoningEffort].(string); ok && effort != "" {
			return true
		}
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
	// in the terminal response event), and the generate reply carries no usage block.
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

// maxFrameBytes bounds a single SSE line within the request's reservation. For the
// chunk shapes the cap is the smaller of the reservation and forceStreamMaxFrameBytes,
// so a frame larger than the reservation is rejected whether or not it reaches 1 MiB;
// with the default per-request cap the reservation stays under 1 MiB and binds first.
// The Responses shape delivers its whole reply in one terminal frame, so its only
// bound is the reservation. The result is clamped to int so a reservation above the
// platform int width cannot wrap negative when it sizes the scanner buffer.
func maxFrameBytes(shape sseShape, reserved int64) int {
	frameCap := reserved
	if shape != sseShapeResponses && frameCap > forceStreamMaxFrameBytes {
		frameCap = forceStreamMaxFrameBytes
	}
	if frameCap > math.MaxInt {
		frameCap = math.MaxInt
	}
	return int(frameCap)
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
	sawDone := false
	for scanner.Scan() {
		payload, ok := ssePayload(scanner.Bytes())
		if !ok {
			continue
		}
		if bytes.Equal(payload, sseDoneMarker) {
			sawDone = true
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
		if errors.Is(err, bufio.ErrTooLong) {
			// The Responses reply arrives in one frame whose cap is the reservation,
			// but the reservation prices only generated tokens, not the request
			// fields (tools, instructions, schema) the completed frame echoes. An
			// overflow there is a coordinator shortfall, not a pod fault, so fall
			// back to the pass-through. For the other shapes an oversized frame is
			// anomalous: a ceiling abort.
			if reassembler.shape == sseShapeResponses {
				return errForceStreamOversize
			}
			return errForceStreamCeiling
		}
		return fmt.Errorf("force-stream: read upstream: %w", err)
	}
	// A clean EOF with neither the [DONE] marker nor a shape terminal means the
	// upstream was cut mid-reply; the reassembly would look complete but omit the
	// tail, so fall back rather than return a truncated reply as a 200.
	if !sawDone && !reassembler.terminated() {
		return errForceStreamTruncated
	}
	return nil
}

// writeForcedResponse marshals the reassembled reply and writes it as one
// non-streaming JSON response. A marshal failure returns a plain error before
// any write, so the server answers a clean 502.
func writeForcedResponse(logger logr.Logger, reqCtx *pipeline.RequestContext, reassembler *sseReassembler) error {
	if reassembler.responsesTerminalMissing() {
		logger.V(logutil.DEFAULT).Info("force-stream responses stream ended without a terminal event; reporting status incomplete", "model", reqCtx.Model)
	}
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
