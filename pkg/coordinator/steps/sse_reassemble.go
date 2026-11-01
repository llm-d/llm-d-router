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
	"encoding/json"
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
// than choices[]; its reply is a response object with an output array. The
// Anthropic Messages API also streams typed events (message_start,
// content_block_delta, message_delta, message_stop); its reply is a message
// object with a content array.
type sseShape int

const (
	sseShapeChat sseShape = iota
	sseShapeText
	sseShapeGenerate
	sseShapeResponses
	sseShapeMessages
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
)

// Anthropic Messages API streaming event types (the data-line "type" value), the
// content-block delta types, and the Messages reply field names that have no shared
// constant. message_start carries the envelope (id, model, role, initial usage)
// nested under "message"; content_block_delta carries a text_delta's text;
// message_delta carries the terminal stop_reason and the authoritative usage;
// message_stop ends the stream (there is no [DONE] marker). A non-text content
// block (tool_use, thinking) makes the reply lossy, so force-streaming falls back.
const (
	messageEventStart        = "message_start"
	messageEventContentStart = "content_block_start"
	messageEventContentDelta = "content_block_delta"
	messageEventDelta        = "message_delta"
	messageEventStop         = "message_stop"

	deltaTypeText = "text_delta"

	objectMessage     = "message"
	fieldMessage      = "message"
	fieldContentBlock = "content_block"
	fieldStopReason   = "stop_reason"
	fieldStopSequence = "stop_sequence"
	partTypeText      = "text"
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
	case reqcommon.APITypeMessages:
		return sseShapeMessages, true
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
// logprobs, and reasoning are not reassembled, nor are a generate
// choice's logprobs. The decode step leaves force-streaming off for a request that
// declares such a field (see forceStreamLossy); a server-side default the request
// does not declare is caught on the frame instead (lossyFrame), and both take the
// pass-through, which preserves the dropped field.
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

	// lossyFrame is set when a chat delta carried a field beyond role and content,
	// which the fold drops. The scan treats it as a fallback so the pass-through
	// keeps the field. It catches a server-side default (reasoning parser, template)
	// that forceStreamLossy cannot predict from the request.
	lossyFrame bool

	// Messages-shape accumulators. The Anthropic reply is assembled from its events
	// rather than captured verbatim: the envelope role, the text folded across
	// content_block_delta events, the terminal stop_reason/stop_sequence, and whether
	// a terminal event (message_delta or message_stop) was seen.
	messageRole      string
	messageText      strings.Builder
	stopReason       any
	stopSequence     any
	messagesTerminal bool

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
	if r.shape == sseShapeMessages {
		r.addMessages(frame)
		return
	}
	r.captureMeta(frame)
	if choices, ok := frame[reqcommon.FieldChoices].([]any); ok {
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
// A text delta accumulates only toward the per-request ceiling (its bytes bound a
// generation that overshoots its token limit before a terminal event arrives). A
// terminal event (completed, incomplete, or failed) captures the response object
// the reply is emitted from, carrying the real status, usage, and error so a
// truncated or failed generation is not reported as completed. A stream that ends
// with no terminal event never sets responseObject, so scanForcedResponse treats it
// as truncated and falls back rather than emitting a reply with no status.
func (r *sseReassembler) addResponses(frame map[string]any) {
	switch frame[reqcommon.FieldType] {
	case responseEventOutputTextDelta:
		// A terminal event already captured the final object; a later delta is
		// redundant, so ignore it rather than regrow the buffer and risk re-tripping
		// the ceiling on a reply the pod completed.
		if r.responseObject != nil {
			return
		}
		if delta, ok := frame[fieldDelta].(string); ok {
			r.contentBytes += int64(len(delta))
		}
	case responseEventCompleted, responseEventIncomplete, responseEventFailed:
		if resp, ok := frame[fieldResponse].(map[string]any); ok {
			r.responseObject = resp
		}
	}
}

// addMessages folds one Anthropic Messages event, keyed by its data-line type.
// message_start carries the envelope (id, model, role, initial usage) nested under
// "message"; content_block_delta's text_delta accumulates the reply text (other
// delta types, and any non-text content_block_start, mark the reply lossy so it
// falls back and keeps the dropped block); message_delta carries the terminal
// stop_reason and authoritative usage; message_stop ends the stream. terminated()
// reports true once message_delta or message_stop is seen, since there is no [DONE].
//
// The fold ignores the content block index and emits one text block, so it
// assumes the reply has a single text block. vLLM opens a new text block only
// after a block of another type, which is non-text and already lossy, so the
// assumption holds for vLLM; a server that splits text across several blocks
// would get them concatenated into one.
func (r *sseReassembler) addMessages(frame map[string]any) {
	switch frame[reqcommon.FieldType] {
	case messageEventStart:
		msg, ok := frame[fieldMessage].(map[string]any)
		if !ok {
			return
		}
		if id, ok := msg["id"].(string); ok && r.id == "" {
			r.id = id
		}
		if model, ok := msg[reqcommon.FieldModel].(string); ok && r.model == "" {
			r.model = model
		}
		if role, ok := msg[reqcommon.FieldRole].(string); ok && r.messageRole == "" {
			r.messageRole = role
		}
		if usage, ok := msg[reqcommon.FieldUsage].(map[string]any); ok {
			r.mergeUsage(usage)
		}
	case messageEventContentStart:
		// A non-text block (tool_use, thinking) is announced here before any delta;
		// the fold keeps only text, so flag the reply lossy.
		if block, ok := frame[fieldContentBlock].(map[string]any); ok {
			if t, ok := block[reqcommon.FieldType].(string); ok && t != partTypeText {
				r.lossyFrame = true
			}
		}
	case messageEventContentDelta:
		delta, ok := frame[fieldDelta].(map[string]any)
		if !ok {
			return
		}
		if t, ok := delta[reqcommon.FieldType].(string); ok && t != deltaTypeText {
			// input_json_delta (tool_use) or thinking_delta: not folded, so lossy.
			r.lossyFrame = true
			return
		}
		if text, ok := delta[partTypeText].(string); ok {
			r.messageText.WriteString(text)
			r.contentBytes += int64(len(text))
		}
	case messageEventDelta:
		if delta, ok := frame[fieldDelta].(map[string]any); ok {
			if sr, ok := delta[fieldStopReason]; ok {
				r.stopReason = sr
			}
			if ss, ok := delta[fieldStopSequence]; ok {
				r.stopSequence = ss
			}
		}
		if usage, ok := frame[reqcommon.FieldUsage].(map[string]any); ok {
			r.mergeUsage(usage)
		}
		r.messagesTerminal = true
	case messageEventStop:
		r.messagesTerminal = true
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

func (r *sseReassembler) choice(index int) *sseChoice {
	c, ok := r.choices[index]
	if !ok {
		c = &sseChoice{index: index}
		r.choices[index] = c
		r.order = append(r.order, index)
	}
	return c
}

// foldChoice appends one streamed choice fragment to its accumulator. An absent
// or null index defaults to 0 so a single-choice stream that omits it still folds
// into one choice. An index that is not a non-negative integer, or is at or beyond
// the declared n, is dropped, so an upstream returning malformed, out-of-range, or
// surplus choices cannot grow the choice map past the reservation.
func (r *sseReassembler) foldChoice(choice map[string]any) {
	index := 0
	if v := choice[reqcommon.FieldIndex]; v != nil {
		n, err := reqcommon.NonNegativeInt(v)
		if err != nil {
			return
		}
		index = n
	}
	if r.maxChoices > 0 && index >= r.maxChoices {
		return
	}
	acc := r.choice(index)

	switch r.shape {
	case sseShapeChat:
		if delta, ok := choice[reqcommon.FieldDelta].(map[string]any); ok {
			if role, ok := delta[reqcommon.FieldRole].(string); ok && role != "" {
				acc.role = role
			}
			if content, ok := delta[reqcommon.FieldContent].(string); ok {
				acc.content.WriteString(content)
				r.contentBytes += int64(len(content))
			}
			// The fold keeps only role and content. Any other key (reasoning,
			// tool_calls, refusal, audio, annotations) would be dropped, so flag the
			// reply lossy and let the scan fall back to the pass-through, which keeps
			// it. This catches a server-side default forceStreamLossy cannot see in
			// the request.
			if deltaHasUnfoldableField(delta) {
				r.lossyFrame = true
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

	if fr, ok := choice[reqcommon.FieldFinishReason]; ok && fr != nil {
		acc.finishReason = fr
		acc.hasFinish = true
	}
}

// deltaHasUnfoldableField reports whether a chat delta carries a field the fold
// drops. Only role and content are folded, so any other key with a non-null value
// is unfoldable (reasoning, tool_calls, refusal, audio, annotations). A
// null-valued key is ignored, since SDKs serialize an absent optional that way and
// it carries no data to lose.
func deltaHasUnfoldableField(delta map[string]any) bool {
	for k, v := range delta {
		if k == reqcommon.FieldRole || k == reqcommon.FieldContent {
			continue
		}
		if v != nil {
			return true
		}
	}
	return false
}

// mergeUsage copies every field a usage block reports into the accumulated
// usage, so a later block overrides an earlier one field by field. The APIs the
// coordinator routes report usage once with every field populated, in the
// trailing include_usage frame; folding per field also tolerates a server that
// reports halves separately without dropping either.
//
// Each field retained for the first time is charged to contentBytes at its
// encoded size, so an upstream streaming usage-only frames with ever new fields
// trips the per-request ceiling rather than growing the map past the
// reservation. A field that repeats is not charged again: a stream that reports
// usage on every chunk keeps a fixed set of keys.
func (r *sseReassembler) mergeUsage(usage map[string]any) {
	if r.usage == nil {
		r.usage = map[string]any{}
	}
	for k, v := range usage {
		if v == nil {
			continue
		}
		if _, seen := r.usage[k]; !seen {
			r.contentBytes += int64(len(k)) + encodedSize(v)
		}
		r.usage[k] = v
	}
}

// encodedSize returns the JSON-encoded size of a decoded value. A value that
// cannot be marshalled is charged its fixed envelope share, which keeps the
// accounting conservative without failing the fold.
func encodedSize(v any) int64 {
	b, err := json.Marshal(v)
	if err != nil {
		return forceStreamPerChoiceOverheadBytes
	}
	return int64(len(b))
}

// bufferedBytes is the running size estimate the per-request ceiling bounds: the
// decoded text held so far for the chat, text, and responses shapes, the
// per-token estimate (len(ids) * forceStreamBytesPerToken) for the generate
// shape, and the retained usage fields (see mergeUsage). The envelope the final
// marshal adds is covered by the reservation's fixed overhead, so it is not
// counted here.
func (r *sseReassembler) bufferedBytes() int64 {
	return r.contentBytes
}

// result builds the single non-streaming reply. Choices come out ordered by
// index. A choice that never carried a finish_reason gets an explicit null, and
// chat choices that never carried a role default to assistant, matching a
// non-streaming reply's shape.
func (r *sseReassembler) result() map[string]any {
	if r.shape == sseShapeResponses {
		// scanForcedResponse returns errForceStreamTruncated unless a terminal event
		// captured the response object, so by here it is always set and emitted
		// verbatim: the object the pod would have returned non-streaming, carrying its
		// own status, usage, and (on failure) error.
		return r.responseObject
	}
	if r.shape == sseShapeMessages {
		return r.messagesResult()
	}

	out := map[string]any{}
	r.writeEnvelope(out)

	indices := append([]int(nil), r.order...)
	sort.Ints(indices)
	choices := make([]any, 0, len(indices))
	for _, idx := range indices {
		choices = append(choices, r.resultChoice(r.choices[idx]))
	}
	out[reqcommon.FieldChoices] = choices

	// The generate endpoint's non-streaming reply carries no usage block, so the
	// forced reply drops any usage its stream reports. Chat and text non-streaming
	// replies do carry usage, so it is folded there.
	if r.usage != nil && r.shape != sseShapeGenerate {
		out[reqcommon.FieldUsage] = r.usage
	}
	return out
}

// messagesResult builds the Anthropic Messages reply from the folded events: the
// envelope (id, type, role, model), one text content block, the terminal
// stop_reason/stop_sequence, and every usage field merged from message_start and
// message_delta. Reassembly runs only for a text-only reply; a non-text block
// flags lossyFrame and falls back before here.
func (r *sseReassembler) messagesResult() map[string]any {
	role := r.messageRole
	if role == "" {
		role = reqcommon.RoleAssistant
	}
	out := map[string]any{
		reqcommon.FieldType: objectMessage,
		reqcommon.FieldRole: role,
		reqcommon.FieldContent: []any{
			map[string]any{reqcommon.FieldType: partTypeText, partTypeText: r.messageText.String()},
		},
	}
	if r.id != "" {
		out["id"] = r.id
	}
	if r.model != "" {
		out[reqcommon.FieldModel] = r.model
	}
	if r.stopReason != nil {
		out[fieldStopReason] = r.stopReason
	}
	if r.stopSequence != nil {
		out[fieldStopSequence] = r.stopSequence
	}
	if r.usage != nil {
		out[reqcommon.FieldUsage] = r.usage
	}
	return out
}

// terminated reports whether a frame signalled the end of generation: a Responses
// terminal event (captured as responseObject), a Messages terminal event
// (message_delta or message_stop), or, for the other shapes, a finish_reason on any
// choice. The scan loop pairs it with the data: [DONE] marker to tell a complete
// stream from one cut off before its end, since the Responses and Messages streams
// close on their terminal event without a [DONE] line.
func (r *sseReassembler) terminated() bool {
	if r.shape == sseShapeResponses {
		return r.responseObject != nil
	}
	if r.shape == sseShapeMessages {
		return r.messagesTerminal
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
		out[reqcommon.FieldPromptLogprobs] = nil
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
		reqcommon.FieldIndex: c.index,
		"logprobs":           nil,
	}
	if c.hasFinish {
		choice[reqcommon.FieldFinishReason] = c.finishReason
	} else {
		choice[reqcommon.FieldFinishReason] = nil
	}

	switch r.shape {
	case sseShapeChat:
		role := c.role
		if role == "" {
			role = reqcommon.RoleAssistant
		}
		choice[reqcommon.FieldMessage] = map[string]any{
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
