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
	"sort"
	"strings"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// sseShape is the response layout a streamed completion folds back into. Chat
// completions carry generated text in a per-chunk delta.content and the
// reassembled reply in a message object; the legacy Completions API carries it
// in choices[].text. vLLM's generate API is token-level: its chunks and reply
// carry choices[].token_ids rather than text, under a request_id envelope with
// no object/id/model fields. The Responses API streams typed events keyed by a
// data-line type field (response.output_text.delta, response.completed) rather
// than choices[]; its reply is a response object with an output array.
type sseShape int

const (
	sseShapeChat sseShape = iota
	sseShapeText
	sseShapeGenerate
	sseShapeResponses
)

// Streaming and non-streaming object values. A chat stream tags each chunk
// chat.completion.chunk; the reassembled reply is a chat.completion. The text
// shape uses text_completion in both, so a stream that omits object still gets
// the right value.
const (
	objectChatCompletion = "chat.completion"
	objectTextCompletion = "text_completion"
)

// Responses API streaming event types (the data-line "type" value) and the
// field names a Responses reply carries that have no shared constant.
// response.output_text.delta streams the generated text; the terminal events
// (response.completed, response.incomplete, response.failed) each carry the
// final response object with its own status, usage, and (on failure) error.
const (
	responseEventOutputTextDelta = "response.output_text.delta"
	responseEventCompleted       = "response.completed"
	responseEventIncomplete      = "response.incomplete"
	responseEventFailed          = "response.failed"
	fieldResponse                = "response"
	fieldDelta                   = "delta"
	fieldStatus                  = "status"
	fieldID                      = "id"
	fieldText                    = "text"
	itemTypeMessage              = "message"
	partTypeOutputText           = "output_text"
	statusIncomplete             = "incomplete"
)

// shapeForAPIType maps the API a request used to the reassembly shape its stream
// folds into. ok is false for an API the force-stream path does not handle; the
// caller then declines to force-stream rather than misparsing the stream as text.
// The decode step rejects such APIs earlier, so this guards against that gate
// loosening without this switch following.
func shapeForAPIType(apiType reqcommon.APIType) (sseShape, bool) {
	switch apiType {
	case reqcommon.APITypeChatCompletions:
		return sseShapeChat, true
	case reqcommon.APITypeCompletions:
		return sseShapeText, true
	case reqcommon.APITypeVLLMGenerate:
		return sseShapeGenerate, true
	case reqcommon.APITypeResponses:
		return sseShapeResponses, true
	default:
		return 0, false
	}
}

// sseChoice accumulates one choice index across the frames that mention it.
// content holds the chat and text shapes' generated text; tokenIDs holds the
// generate shape's token stream.
type sseChoice struct {
	index        int
	role         string
	content      strings.Builder
	tokenIDs     []any
	finishReason any
	hasFinish    bool
}

// sseReassembler folds the data frames of one streamed completion into the
// single non-streaming reply the client asked for. It captures the envelope
// fields once (first non-empty wins, since every chunk repeats them), appends
// each choice's generated text or tokens by index so n>1 responses stay
// separate, and merges the usage block the include_usage trailing frame carries.
//
// Only the fields a non-streaming reply needs are folded: text content or token
// ids, role, finish_reason, and usage. A chat delta's tool_calls, function_call,
// logprobs, and reasoning_content are not reassembled; the decode step leaves
// force-streaming off for a chat or text request that declares tools, functions,
// logprobs, or reasoning (see forceStreamLossy), so such a reply takes the
// pass-through instead.
type sseReassembler struct {
	shape sseShape
	// maxChoices caps how many choice indices fold, matching the client's declared
	// n so an upstream reporting more choices than reserved cannot grow memory past
	// the reservation. Zero means no cap.
	maxChoices int

	id                string
	model             string
	object            string
	systemFingerprint string
	requestID         string
	created           any
	hasCreated        bool

	choices map[int]*sseChoice
	order   []int

	usage map[string]any

	// responseObject is the terminal response object captured from a terminal
	// Responses event (completed, incomplete, or failed), emitted verbatim as the
	// reply so its status, usage, and error reach the client unchanged.
	responseObject map[string]any

	contentBytes int64
}

func newSSEReassembler(shape sseShape, maxChoices int) *sseReassembler {
	return &sseReassembler{shape: shape, maxChoices: maxChoices, choices: map[int]*sseChoice{}}
}

// add folds one parsed SSE data frame. Frames with no choices (the trailing
// usage frame) still contribute their usage block.
func (r *sseReassembler) add(frame map[string]any) {
	if r.shape == sseShapeResponses {
		r.addResponses(frame)
		return
	}
	r.captureMeta(frame)
	if choices, ok := frame["choices"].([]any); ok {
		for _, c := range choices {
			if choice, ok := c.(map[string]any); ok {
				r.foldChoice(choice)
			}
		}
	}
	if usage, ok := frame[reqcommon.FieldUsage].(map[string]any); ok {
		r.mergeUsage(usage)
	}
}

// addResponses folds one Responses API event, keyed by its data-line type.
// Text deltas accumulate for the per-request ceiling and for the no-terminal
// fallback reply; a terminal event (completed, incomplete, or failed) captures
// its response object, carrying the real status, usage, and error so a truncated
// or failed generation is not reported as completed. Other events
// (response.created, response.in_progress) seed id and model for the fallback.
func (r *sseReassembler) addResponses(frame map[string]any) {
	switch frame[reqcommon.FieldType] {
	case responseEventOutputTextDelta:
		// A terminal event already captured the final object and released the
		// accumulators; a later delta is redundant, so ignore it rather than regrow
		// the buffer and risk re-tripping the ceiling on a reply the pod completed.
		if r.responseObject != nil {
			return
		}
		if delta, ok := frame[fieldDelta].(string); ok {
			r.choice(0).content.WriteString(delta)
			r.contentBytes += int64(len(delta))
		}
	case responseEventCompleted, responseEventIncomplete, responseEventFailed:
		if resp, ok := frame[fieldResponse].(map[string]any); ok {
			r.responseObject = resp
			// The accumulated deltas fed only the ceiling and the no-terminal
			// fallback; the terminal object supersedes them, so release them
			// rather than hold the content twice.
			r.choices = map[int]*sseChoice{}
			r.order = nil
		}
	default:
		if resp, ok := frame[fieldResponse].(map[string]any); ok {
			r.captureResponseMeta(resp)
		}
	}
}

// captureResponseMeta records the id and model the fallback reply needs, taken
// from the response envelope an early event carries (first non-empty wins).
func (r *sseReassembler) captureResponseMeta(resp map[string]any) {
	if r.id == "" {
		if id, ok := resp[fieldID].(string); ok {
			r.id = id
		}
	}
	if r.model == "" {
		if model, ok := resp[reqcommon.FieldModel].(string); ok {
			r.model = model
		}
	}
}

// captureMeta records the envelope fields the first chunk to carry each one
// provides. object is skipped: the reassembled object is fixed per shape, not
// the chunk's chat.completion.chunk. The generate shape carries none of these
// and folds a request_id instead.
func (r *sseReassembler) captureMeta(frame map[string]any) {
	if r.shape == sseShapeGenerate {
		if r.requestID == "" {
			if id, ok := frame["request_id"].(string); ok {
				r.requestID = id
			}
		}
		return
	}
	if r.id == "" {
		if id, ok := frame["id"].(string); ok {
			r.id = id
		}
	}
	if r.model == "" {
		if model, ok := frame[reqcommon.FieldModel].(string); ok {
			r.model = model
		}
	}
	if r.systemFingerprint == "" {
		if fp, ok := frame["system_fingerprint"].(string); ok {
			r.systemFingerprint = fp
		}
	}
	if r.shape == sseShapeText && r.object == "" {
		if object, ok := frame["object"].(string); ok {
			r.object = object
		}
	}
	if !r.hasCreated {
		if created, ok := frame["created"]; ok && created != nil {
			r.created = created
			r.hasCreated = true
		}
	}
}

// numField reads a numeric choice index. JSON decodes it to float64; a producer
// that supplies a Go int is tolerated too.
func numField(v any) (int, bool) {
	switch n := v.(type) {
	case float64:
		return int(n), true
	case int:
		return n, true
	default:
		return 0, false
	}
}

func (r *sseReassembler) choice(index int) *sseChoice {
	c, ok := r.choices[index]
	if !ok {
		c = &sseChoice{index: index}
		r.choices[index] = c
		r.order = append(r.order, index)
	}
	return c
}

// foldChoice appends one streamed choice fragment to its accumulator. The index
// defaults to 0 so a single-choice stream that omits it still folds into one
// choice. An index that is negative or at or beyond the declared n is dropped, so
// an upstream returning out-of-range or surplus choices cannot grow the choice map
// past the reservation.
func (r *sseReassembler) foldChoice(choice map[string]any) {
	index := 0
	if n, ok := numField(choice["index"]); ok {
		index = n
	}
	if index < 0 || (r.maxChoices > 0 && index >= r.maxChoices) {
		return
	}
	acc := r.choice(index)

	switch r.shape {
	case sseShapeChat:
		if delta, ok := choice["delta"].(map[string]any); ok {
			if role, ok := delta[reqcommon.FieldRole].(string); ok && role != "" {
				acc.role = role
			}
			if content, ok := delta[reqcommon.FieldContent].(string); ok {
				acc.content.WriteString(content)
				r.contentBytes += int64(len(content))
			}
		}
	case sseShapeGenerate:
		if ids, ok := choice["token_ids"].([]any); ok {
			acc.tokenIDs = append(acc.tokenIDs, ids...)
			// Account each buffered token at the per-token byte estimate the
			// reservation uses, so an upstream that ignores max_tokens trips the
			// per-request ceiling just as the text shape does.
			r.contentBytes += int64(len(ids)) * forceStreamBytesPerToken
		}
	case sseShapeText:
		if text, ok := choice["text"].(string); ok {
			acc.content.WriteString(text)
			r.contentBytes += int64(len(text))
		}
	}

	if fr, ok := choice["finish_reason"]; ok && fr != nil {
		acc.finishReason = fr
		acc.hasFinish = true
	}
}

// mergeUsage copies every field a usage block reports into the accumulated
// usage, so a later block overrides an earlier one field by field. The APIs the
// coordinator routes report usage once with every field populated, in the
// trailing include_usage frame; folding per field also tolerates a server that
// reports halves separately without dropping either.
func (r *sseReassembler) mergeUsage(usage map[string]any) {
	if r.usage == nil {
		r.usage = map[string]any{}
	}
	for k, v := range usage {
		if v != nil {
			r.usage[k] = v
		}
	}
}

// bufferedBytes is the running size estimate the per-request ceiling bounds: the
// decoded text held so far for the chat, text, and responses shapes, and the
// per-token estimate (len(ids) * forceStreamBytesPerToken) for the generate
// shape. The envelope the final marshal adds is covered by the reservation's
// fixed overhead, so it is not counted here.
func (r *sseReassembler) bufferedBytes() int64 {
	return r.contentBytes
}

// result builds the single non-streaming reply. Choices come out ordered by
// index. A choice that never carried a finish_reason gets an explicit null, and
// chat choices that never carried a role default to assistant, matching a
// non-streaming reply's shape.
func (r *sseReassembler) result() map[string]any {
	if r.shape == sseShapeResponses {
		return r.responsesResult()
	}

	out := map[string]any{}
	r.writeEnvelope(out)

	indices := append([]int(nil), r.order...)
	sort.Ints(indices)
	choices := make([]any, 0, len(indices))
	for _, idx := range indices {
		choices = append(choices, r.resultChoice(r.choices[idx]))
	}
	out["choices"] = choices

	// The generate endpoint's non-streaming reply carries no usage block, so the
	// forced reply drops the one its stream_options.include_usage frame supplied.
	// Chat and text non-streaming replies do carry usage, so it is folded there.
	if r.usage != nil && r.shape != sseShapeGenerate {
		out[reqcommon.FieldUsage] = r.usage
	}
	return out
}

// responsesResult returns the Responses reply. A terminal event (completed,
// incomplete, or failed) carries the response object the pod would have returned
// non-streaming, so it is emitted verbatim. When no terminal event arrived, a
// minimal response object is assembled from the accumulated output_text deltas and
// marked status incomplete: the collected text still reaches the client, but the
// stream never confirmed completion, so it is not reported as completed.
func (r *sseReassembler) responsesResult() map[string]any {
	if r.responseObject != nil {
		return r.responseObject
	}

	content := ""
	if c, ok := r.choices[0]; ok {
		content = c.content.String()
	}
	out := map[string]any{
		fieldStatus: statusIncomplete,
		reqcommon.FieldOutput: []any{
			map[string]any{
				reqcommon.FieldType: itemTypeMessage,
				reqcommon.FieldRole: reqcommon.RoleAssistant,
				reqcommon.FieldContent: []any{
					map[string]any{reqcommon.FieldType: partTypeOutputText, fieldText: content},
				},
			},
		},
	}
	if r.id != "" {
		out[fieldID] = r.id
	}
	if r.model != "" {
		out[reqcommon.FieldModel] = r.model
	}
	return out
}

// responsesTerminalMissing reports a Responses stream that ended with no terminal
// event. The reply was synthesized from the deltas and marked incomplete, so its
// completion is unconfirmed; the caller logs it to keep the gap diagnosable.
func (r *sseReassembler) responsesTerminalMissing() bool {
	return r.shape == sseShapeResponses && r.responseObject == nil
}

// terminated reports whether a frame signalled the end of generation: a Responses
// terminal event (captured as responseObject) or, for the other shapes, a
// finish_reason on any choice. The scan loop pairs it with the data: [DONE] marker
// to tell a complete stream from one cut off before its end, since the Responses
// stream may close on its terminal event without a [DONE] line.
func (r *sseReassembler) terminated() bool {
	if r.shape == sseShapeResponses {
		return r.responseObject != nil
	}
	for _, c := range r.choices {
		if c.hasFinish {
			return true
		}
	}
	return false
}

// writeEnvelope sets the top-level reply fields for the shape. The chat and text
// shapes share the OpenAI envelope (id, model, object, ...). The generate shape
// instead carries request_id, prompt_logprobs, and kv_transfer_params, and no
// object. kv_transfer_params is null in the terminal decode reply (the remote
// handoff it would carry is consumed before the client-facing reply); emitting
// it as null keeps the forced reply identical to the worker's own non-streaming
// generate reply.
func (r *sseReassembler) writeEnvelope(out map[string]any) {
	if r.shape == sseShapeGenerate {
		if r.requestID != "" {
			out["request_id"] = r.requestID
		}
		out["prompt_logprobs"] = nil
		out[reqcommon.FieldKVTransferParams] = nil
		return
	}
	if r.id != "" {
		out["id"] = r.id
	}
	if r.model != "" {
		out[reqcommon.FieldModel] = r.model
	}
	if r.systemFingerprint != "" {
		out["system_fingerprint"] = r.systemFingerprint
	}
	if r.hasCreated {
		out["created"] = r.created
	}
	out["object"] = r.resultObject()
}

func (r *sseReassembler) resultObject() string {
	if r.shape == sseShapeChat {
		return objectChatCompletion
	}
	if r.object != "" {
		return r.object
	}
	return objectTextCompletion
}

func (r *sseReassembler) resultChoice(c *sseChoice) map[string]any {
	choice := map[string]any{
		"index":    c.index,
		"logprobs": nil,
	}
	if c.hasFinish {
		choice["finish_reason"] = c.finishReason
	} else {
		choice["finish_reason"] = nil
	}

	switch r.shape {
	case sseShapeChat:
		role := c.role
		if role == "" {
			role = reqcommon.RoleAssistant
		}
		choice["message"] = map[string]any{
			reqcommon.FieldRole:    role,
			reqcommon.FieldContent: c.content.String(),
		}
	case sseShapeGenerate:
		tokens := c.tokenIDs
		if tokens == nil {
			tokens = []any{}
		}
		choice["token_ids"] = tokens
	default:
		choice["text"] = c.content.String()
	}
	return choice
}
