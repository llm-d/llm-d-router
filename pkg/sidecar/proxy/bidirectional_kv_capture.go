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

package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/felixge/httpsnoop"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// maxCapturedResponseBytes bounds how much of one decode response the capture
// reads. A larger response is forwarded unchanged and never cached.
const maxCapturedResponseBytes = 16 << 20

// decodeCapture observes a Chat Completions decode response while it streams to
// the client and records what bidirectional KV transfer needs from it: the
// decode engine's kv_transfer_params and the assistant message it generated.
// Bytes are forwarded untouched and unbuffered, so SSE keeps flowing.
type decodeCapture struct {
	header http.Header

	started  bool
	sse      bool
	seen     int          // response bytes observed so far
	body     bytes.Buffer // non-streaming JSON body
	pending  []byte       // SSE bytes after the last newline
	overflow bool
	// unusable is set when the response cannot be reduced to one assistant
	// message: more than one choice, a data frame that does not parse, or
	// reasoning output that a chat template may render differently on the next turn.
	unusable bool

	kvParams map[string]any
	content  strings.Builder
	calls    []*capturedToolCall
	callByIx map[int]*capturedToolCall
}

// capturedToolCall is a tool call, or a fragment of one in a stream. index is
// the fragment's slot in a stream and -1 when the wire form carries none.
type capturedToolCall struct {
	index               int
	id, name, arguments string
}

// capturedResponse is the subset of a Chat Completions response or stream
// chunk that the capture reads.
type capturedResponse struct {
	KVTransferParams map[string]any   `json:"kv_transfer_params"`
	Choices          []capturedChoice `json:"choices"`
}

type capturedChoice struct {
	Index   int              `json:"index"`
	Delta   *capturedMessage `json:"delta"`
	Message *capturedMessage `json:"message"`
}

type capturedMessage struct {
	Content   *string            `json:"content"`
	ToolCalls []capturedToolCall `json:"tool_calls"`
	// Reasoning is the model's reasoning text under either name vLLM has used.
	ReasoningContent *string `json:"reasoning_content"`
	Reasoning        *string `json:"reasoning"`
}

func (m *capturedMessage) hasReasoning() bool {
	return (m.ReasoningContent != nil && *m.ReasoningContent != "") || (m.Reasoning != nil && *m.Reasoning != "")
}

// UnmarshalJSON reads a tool call in the wire shape. A streamed delta carries
// an index; a complete message does not.
func (t *capturedToolCall) UnmarshalJSON(b []byte) error {
	var wire struct {
		Index    *int   `json:"index"`
		ID       string `json:"id"`
		Function struct {
			Name      string `json:"name"`
			Arguments string `json:"arguments"`
		} `json:"function"`
	}
	if err := json.Unmarshal(b, &wire); err != nil {
		return err
	}
	t.index = -1
	if wire.Index != nil {
		t.index = *wire.Index
	}
	t.id, t.name, t.arguments = wire.ID, wire.Function.Name, wire.Function.Arguments
	return nil
}

// writerFunc adapts a function to io.Writer.
type writerFunc func(p []byte) (int, error)

func (f writerFunc) Write(p []byte) (int, error) { return f(p) }

// newDecodeCapture wraps w so every byte written through it is also observed
// by the returned capture. httpsnoop keeps w's http.Flusher, http.Hijacker and
// io.ReaderFrom behavior, and the ReadFrom hook routes bulk copies through
// the same observation.
func newDecodeCapture(w http.ResponseWriter) (http.ResponseWriter, *decodeCapture) {
	c := &decodeCapture{header: w.Header()}
	wrapped := httpsnoop.Wrap(w, httpsnoop.Hooks{
		Write: func(next httpsnoop.WriteFunc) httpsnoop.WriteFunc {
			return func(p []byte) (int, error) {
				c.observe(p)
				return next(p)
			}
		},
		ReadFrom: func(httpsnoop.ReadFromFunc) httpsnoop.ReadFromFunc {
			return func(src io.Reader) (int64, error) {
				return io.Copy(writerFunc(func(p []byte) (int, error) {
					c.observe(p)
					return w.Write(p)
				}), src)
			}
		},
	})
	return wrapped, c
}

func (c *decodeCapture) observe(p []byte) {
	if !c.started {
		c.started = true
		c.sse = strings.HasPrefix(c.header.Get("Content-Type"), "text/event-stream")
	}
	if c.overflow {
		return
	}
	c.seen += len(p)
	if c.seen > maxCapturedResponseBytes {
		c.overflow = true
		c.body.Reset()
		c.pending = nil
		return
	}
	if !c.sse {
		c.body.Write(p)
		return
	}
	c.pending = append(c.pending, p...)
	for {
		i := bytes.IndexByte(c.pending, '\n')
		if i < 0 {
			break
		}
		c.absorbLine(c.pending[:i])
		c.pending = c.pending[i+1:]
	}
}

// absorbLine reads one SSE line. Only data frames carry content; comments and
// other fields are ignored.
func (c *decodeCapture) absorbLine(line []byte) {
	line = bytes.TrimRight(line, "\r")
	data, ok := bytes.CutPrefix(line, []byte("data:"))
	if !ok {
		return
	}
	data = bytes.TrimSpace(data)
	if len(data) == 0 || string(data) == reqcommon.SSEDoneMarker {
		return
	}
	c.absorb(data, true)
}

// absorb folds one response body or stream chunk into the capture.
func (c *decodeCapture) absorb(data []byte, streaming bool) {
	var resp capturedResponse
	dec := json.NewDecoder(bytes.NewReader(data))
	// Keep numbers as written so block IDs and ports replay verbatim.
	dec.UseNumber()
	if err := dec.Decode(&resp); err != nil {
		c.unusable = true
		return
	}
	if resp.KVTransferParams != nil {
		c.kvParams = resp.KVTransferParams
	}
	for _, choice := range resp.Choices {
		if choice.Index != 0 {
			c.unusable = true
			return
		}
		switch {
		case streaming && choice.Delta != nil:
			c.absorbDelta(choice.Delta)
		case !streaming && choice.Message != nil:
			c.absorbMessage(choice.Message)
		}
	}
}

func (c *decodeCapture) absorbMessage(m *capturedMessage) {
	if m.hasReasoning() {
		c.unusable = true
	}
	if m.Content != nil {
		c.content.WriteString(*m.Content)
	}
	for i := range m.ToolCalls {
		tc := m.ToolCalls[i]
		c.calls = append(c.calls, &tc)
	}
}

// absorbDelta folds one streamed delta in. Tool calls arrive in fragments keyed
// by index: the id and name in the first, the arguments spread across the rest.
func (c *decodeCapture) absorbDelta(d *capturedMessage) {
	if d.hasReasoning() {
		c.unusable = true
	}
	if d.Content != nil {
		c.content.WriteString(*d.Content)
	}
	for i := range d.ToolCalls {
		frag := d.ToolCalls[i]
		ix := frag.index
		if ix < 0 {
			ix = i
		}
		if c.callByIx == nil {
			c.callByIx = make(map[int]*capturedToolCall)
		}
		call, ok := c.callByIx[ix]
		if !ok {
			call = &capturedToolCall{}
			c.callByIx[ix] = call
			c.calls = append(c.calls, call)
		}
		if call.id == "" {
			call.id = frag.id
		}
		if call.name == "" {
			call.name = frag.name
		}
		call.arguments += frag.arguments
	}
}

// result returns the decode engine's replayable kv_transfer_params and the
// assistant message it generated. ok is false when the response cannot be
// cached: no usable params, an unparsable or multi-choice response, or a body
// over the capture limit.
func (c *decodeCapture) result() (params, reply map[string]any, ok bool) {
	if c.sse {
		if len(c.pending) > 0 {
			c.absorbLine(c.pending)
			c.pending = nil
		}
	} else if c.body.Len() > 0 {
		body := c.body.Bytes()
		c.body = bytes.Buffer{}
		c.absorb(body, false)
	}
	if c.unusable || c.overflow {
		return nil, nil, false
	}
	params, ok = completeKVParams(c.kvParams)
	if !ok {
		return nil, nil, false
	}
	reply = map[string]any{reqcommon.FieldRole: roleAssistant}
	if content := c.content.String(); content != "" {
		reply[reqcommon.FieldContent] = content
	}
	if len(c.calls) > 0 {
		calls := make([]any, len(c.calls))
		for i, call := range c.calls {
			calls[i] = map[string]any{
				toolCallFieldID:   call.id,
				toolCallFieldType: toolCallTypeFunction,
				toolCallFieldFunction: map[string]any{
					toolCallFieldName:      call.name,
					toolCallFieldArguments: call.arguments,
				},
			}
		}
		reply[messageFieldToolCalls] = calls
	}
	return params, reply, true
}
