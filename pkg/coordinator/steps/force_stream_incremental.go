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
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/go-logr/logr"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"

	"github.com/llm-d/llm-d-router/pkg/coordinator/common/httplog"
	"github.com/llm-d/llm-d-router/pkg/coordinator/gateway"
	coordmetrics "github.com/llm-d/llm-d-router/pkg/coordinator/metrics"
	"github.com/llm-d/llm-d-router/pkg/coordinator/pipeline"
)

// canStreamIncrementally reports whether a forced request can be written to the
// client incrementally rather than buffered whole. The chat and text
// shapes with a single choice qualify: their reply is one linear content string,
// so it can be emitted as it folds. Multi-choice replies interleave by index,
// and the generate and Responses shapes are not a flat string, so those take the
// buffered path.
func canStreamIncrementally(shape sseShape, reqCtx *pipeline.RequestContext) bool {
	if shape != sseShapeChat && shape != sseShapeText {
		return false
	}
	apiType := reqcommon.DetectAPIType(reqCtx.OriginalPath)
	return reqcommon.OutputChoiceCount(reqCtx.Body, apiType) == 1
}

// executeForceStreamIncremental forces streaming upstream and writes the single
// non-streaming reply to the client as the frames arrive, holding only one frame
// at a time. It reserves no budget: memory is bounded by construction, so unlike
// the buffered path it force-streams a request with no declared token limit.
//
// The trade is error semantics. Nothing is on the wire until the first content
// byte is flushed; before that a transport error or a 4xx/5xx upstream status is
// a clean failure. After that the response is committed, so a mid-stream upstream
// fault can only truncate the connection, the same degraded mode as the
// pass-through.
func (s *DecodeStep) executeForceStreamIncremental(ctx context.Context, logger logr.Logger, reqCtx *pipeline.RequestContext, shape sseShape) error {
	bodyBytes, headers, err := buildForceStreamRequest(reqCtx, shape)
	if err != nil {
		return err
	}

	logger.V(logutil.DEFAULT).Info("force-streaming request (incremental)", "path", reqCtx.OriginalPath)
	if v := logger.V(logutil.DEBUG); v.Enabled() {
		v.Info("force-stream request body", "method", "POST", "path", reqCtx.OriginalPath, "bodyLen", len(bodyBytes), "headers", httplog.RedactedHeaders(headers))
	}

	call := coordmetrics.StartUpstreamCall(coordmetrics.UpstreamDecode)
	resp, err := s.gwClient.Post(ctx, reqCtx.OriginalPath, bodyBytes, headers)
	call.Done()
	if err != nil {
		return fmt.Errorf("%s: force-stream request: %w", DecodeStepName, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= http.StatusBadRequest {
		return forwardUpstreamError(reqCtx, resp)
	}

	return streamForcedResponse(logger, reqCtx, resp.Body, shape)
}

// streamForcedResponse scans the forced upstream stream and writes the reply
// incrementally. It omits Content-Length so net/http uses chunked encoding, and
// clears the write deadline for the slow-client drain.
func streamForcedResponse(logger logr.Logger, reqCtx *pipeline.RequestContext, body io.Reader, shape sseShape) error {
	em := newIncrementalEmitter(reqCtx.ResponseWriter, shape, reqCtx.Model)
	em.clearWriteDeadline()

	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 0, forceStreamScanStartBytes), forceStreamMaxFrameBytes)
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
			return em.failOrTruncate(logger, fmt.Errorf("force-stream: parse frame: %w", err))
		}
		if err := em.add(frame); err != nil {
			return em.failOrTruncate(logger, err)
		}
	}
	if err := scanner.Err(); err != nil {
		return em.failOrTruncate(logger, fmt.Errorf("force-stream: read upstream: %w", err))
	}
	return em.finish(logger)
}

// incrementalEmitter writes one non-streaming reply from a forced stream without
// buffering the whole body. It captures the envelope fields until the first
// content byte, then writes the JSON prefix, streams each escaped content delta,
// and writes the suffix (finish_reason, usage) once the stream ends.
type incrementalEmitter struct {
	w        http.ResponseWriter
	rc       *http.ResponseController
	shape    sseShape
	reqModel string

	started bool

	id                string
	model             string
	systemFingerprint string
	object            string
	role              string
	created           any
	hasCreated        bool

	finishReason any
	hasFinish    bool
	usage        map[string]any
}

func newIncrementalEmitter(w http.ResponseWriter, shape sseShape, model string) *incrementalEmitter {
	return &incrementalEmitter{w: w, rc: http.NewResponseController(w), shape: shape, reqModel: model}
}

// clearWriteDeadline disables the server write timeout for this response so a
// slow client does not cut the drain short. Writers that do not support it
// (the test recorder) report ErrNotSupported, which is ignored.
func (e *incrementalEmitter) clearWriteDeadline() {
	_ = e.rc.SetWriteDeadline(time.Time{})
}

// add folds one frame: it captures envelope and trailing fields and writes any
// content delta. The prefix is written lazily on the first content byte.
func (e *incrementalEmitter) add(frame map[string]any) error {
	e.captureMeta(frame)

	if choices, ok := frame["choices"].([]any); ok && len(choices) > 0 {
		if c0, ok := choices[0].(map[string]any); ok {
			if fr, ok := c0["finish_reason"]; ok && fr != nil {
				e.finishReason = fr
				e.hasFinish = true
			}
		}
	}
	if usage, ok := frame["usage"].(map[string]any); ok {
		e.mergeUsage(usage)
	}

	content, ok := e.frameContent(frame)
	if !ok || content == "" {
		return nil
	}
	if err := e.ensureStarted(); err != nil {
		return err
	}
	esc, _ := json.Marshal(content)
	if _, err := e.w.Write(esc[1 : len(esc)-1]); err != nil {
		return err
	}
	_ = e.rc.Flush()
	return nil
}

// captureMeta records the envelope fields the reply needs, first non-empty wins,
// matching the buffered reassembler. The chat role lives in a delta; the text
// shape may carry its own object value.
func (e *incrementalEmitter) captureMeta(frame map[string]any) {
	if e.id == "" {
		if v, ok := frame["id"].(string); ok {
			e.id = v
		}
	}
	if e.model == "" {
		if v, ok := frame["model"].(string); ok {
			e.model = v
		}
	}
	if e.systemFingerprint == "" {
		if v, ok := frame["system_fingerprint"].(string); ok {
			e.systemFingerprint = v
		}
	}
	if !e.hasCreated {
		if v, ok := frame["created"]; ok && v != nil {
			e.created = v
			e.hasCreated = true
		}
	}
	if e.shape == sseShapeText && e.object == "" {
		if v, ok := frame["object"].(string); ok {
			e.object = v
		}
	}
	if e.shape == sseShapeChat && e.role == "" {
		if c0 := firstChoice(frame); c0 != nil {
			if delta, ok := c0["delta"].(map[string]any); ok {
				if role, ok := delta["role"].(string); ok {
					e.role = role
				}
			}
		}
	}
}

// frameContent returns the generated text a frame carries for the shape.
func (e *incrementalEmitter) frameContent(frame map[string]any) (string, bool) {
	c0 := firstChoice(frame)
	if c0 == nil {
		return "", false
	}
	switch e.shape {
	case sseShapeChat:
		if delta, ok := c0["delta"].(map[string]any); ok {
			if content, ok := delta["content"].(string); ok {
				return content, true
			}
		}
	case sseShapeText:
		if text, ok := c0["text"].(string); ok {
			return text, true
		}
	}
	return "", false
}

func (e *incrementalEmitter) mergeUsage(usage map[string]any) {
	if e.usage == nil {
		e.usage = map[string]any{}
	}
	for k, v := range usage {
		if v != nil {
			e.usage[k] = v
		}
	}
}

// ensureStarted writes the response head and the JSON prefix exactly once,
// before any content.
func (e *incrementalEmitter) ensureStarted() error {
	if e.started {
		return nil
	}
	e.w.Header().Set(gateway.ContentTypeHeader, reqcommon.ContentTypeJSON)
	e.w.WriteHeader(http.StatusOK)
	// The response is committed once the status is written, so mark started before
	// the body write: a failed prefix write must truncate, not be mistaken for an
	// uncommitted request that the server could still answer with a clean 5xx.
	e.started = true
	_, err := e.w.Write(e.prefix())
	return err
}

// finish writes the suffix once the stream ends. An empty completion still gets
// a well-formed reply, since ensureStarted writes the prefix here if no content
// arrived.
func (e *incrementalEmitter) finish(logger logr.Logger) error {
	if err := e.ensureStarted(); err != nil {
		return e.failOrTruncate(logger, err)
	}
	if _, err := e.w.Write(e.suffix()); err != nil {
		logger.Error(err, "force-stream client write incomplete")
		return nil
	}
	_ = e.rc.Flush()
	coordmetrics.IncForceStreamTotal(e.reqModel, coordmetrics.ForceStreamResultForced)
	logger.V(logutil.DEFAULT).Info("force-stream complete (incremental)")
	return nil
}

// failOrTruncate decides what a mid-stream error means. Before the first byte is
// written the request is uncommitted, so the error propagates and the server
// answers a clean 5xx. After the first write the response is committed, so the
// fault can only end the connection; it is logged and reported as handled.
func (e *incrementalEmitter) failOrTruncate(logger logr.Logger, err error) error {
	if !e.started {
		return fmt.Errorf("%s: %w", DecodeStepName, err)
	}
	logger.Error(err, "force-stream truncated after commit")
	return nil
}

// prefix is the reply JSON up to the open quote of the content string.
func (e *incrementalEmitter) prefix() []byte {
	var b bytes.Buffer
	b.WriteByte('{')
	e.writeEnvelopeFields(&b)
	if e.shape == sseShapeChat {
		role := e.role
		if role == "" {
			role = roleAssistant
		}
		roleJSON, _ := json.Marshal(role)
		b.WriteString(`"choices":[{"index":0,"logprobs":null,"message":{"role":`)
		b.Write(roleJSON)
		b.WriteString(`,"content":"`)
	} else {
		b.WriteString(`"choices":[{"index":0,"logprobs":null,"text":"`)
	}
	return b.Bytes()
}

// suffix closes the content string and writes the trailing reply fields.
func (e *incrementalEmitter) suffix() []byte {
	var b bytes.Buffer
	if e.shape == sseShapeChat {
		b.WriteString(`"}`) // close content string and message object
	} else {
		b.WriteByte('"') // close content string
	}
	b.WriteString(`,"finish_reason":`)
	var fr any
	if e.hasFinish {
		fr = e.finishReason
	}
	frJSON, _ := json.Marshal(fr)
	b.Write(frJSON)
	b.WriteString(`}]`) // close choice object and choices array
	if e.usage != nil {
		b.WriteString(`,"usage":`)
		usageJSON, _ := json.Marshal(e.usage)
		b.Write(usageJSON)
	}
	b.WriteByte('}')
	return b.Bytes()
}

// writeEnvelopeFields appends the top-level fields the reply shares, each as a
// trailing-comma pair so the choices array can follow without a separator. Only
// non-empty captured fields are emitted, matching the buffered reassembler.
func (e *incrementalEmitter) writeEnvelopeFields(b *bytes.Buffer) {
	if e.id != "" {
		appendField(b, "id", e.id)
	}
	if e.model != "" {
		appendField(b, "model", e.model)
	}
	if e.systemFingerprint != "" {
		appendField(b, "system_fingerprint", e.systemFingerprint)
	}
	if e.hasCreated {
		appendField(b, "created", e.created)
	}
	object := objectChatCompletion
	if e.shape == sseShapeText {
		object = e.object
		if object == "" {
			object = objectTextCompletion
		}
	}
	appendField(b, "object", object)
}

// appendField writes a "key":value, pair with the value JSON-encoded.
func appendField(b *bytes.Buffer, key string, v any) {
	b.WriteByte('"')
	b.WriteString(key)
	b.WriteString(`":`)
	vb, _ := json.Marshal(v)
	b.Write(vb)
	b.WriteByte(',')
}

// firstChoice returns the first element of a frame's choices array, or nil.
func firstChoice(frame map[string]any) map[string]any {
	choices, ok := frame["choices"].([]any)
	if !ok || len(choices) == 0 {
		return nil
	}
	c0, _ := choices[0].(map[string]any)
	return c0
}
