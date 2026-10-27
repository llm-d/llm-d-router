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
	"fmt"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// generateOnlyResponseFields are the generate-envelope fields that have no
// place in a chat completions response: the KV and EC handoff descriptors are
// coordination state between the phases, and the rest describe the prompt and
// the engine rather than the answer. They are dropped rather than passed
// through, since a client reading the OpenAI schema has nowhere to put them.
var generateOnlyResponseFields = []string{
	reqcommon.FieldRequestID,
	reqcommon.FieldKVTransferParams,
	reqcommon.FieldECTransferParams,
	"prompt_logprobs",
	"prompt_token_id_logprobs",
	"prompt_token_ids",
	"mm_placeholders",
	"metrics",
	reqcommon.FieldOutputMode,
}

// generateOnlyChoiceFields are the per-choice counterparts: the token IDs the
// text or message already spells out, and the engine-level detail a chat
// completions choice does not declare.
var generateOnlyChoiceFields = []string{
	"token_ids",
	"routed_experts",
	"sampling_mask",
	"weight_versions",
}

// wrapChatCompletion rewrites a vLLM generate response into a chat completions
// response. vLLM serves output_mode on the generate envelope rather than
// returning an OpenAI one (the response schema would otherwise depend on a
// request field), so a client that called /v1/chat/completions and had its
// request rewritten to the generate route needs the envelope put back.
//
// It carries over what the two schemas share, drops the generate-only fields,
// and labels the result. A choice's message is moved across as it arrived, so
// the reasoning and tool_calls an output_mode of derender produces survive
// without this package modeling them.
//
// A response whose choices carry text rather than a message (output_mode text)
// is given the message a chat completions client reads, built from that text.
//
// requestID seeds the id field: the generate response's own request_id is the
// one the coordinator sent, so the two agree, and a response that omitted it
// still names the request the client asked about.
//
// wantMode is the output_mode the leg asked for. A worker predating the field
// ignores it and answers with token IDs only, which this rewrite would hand to
// the client as an empty message, so a response not echoing the mode is an
// error rather than an answer.
func wrapChatCompletion(generate []byte, requestID, wantMode string) ([]byte, error) {
	var resp map[string]any
	if err := json.Unmarshal(generate, &resp); err != nil {
		return nil, fmt.Errorf("decoding generate response: %w", err)
	}

	// Absent means tokens, the default of a worker that knows the field and of
	// one that does not.
	gotMode, _ := resp[reqcommon.FieldOutputMode].(string)
	if gotMode != wantMode {
		return nil, fmt.Errorf("worker answered output_mode %q, want %q: it carries no text to answer with",
			gotMode, wantMode)
	}

	id, _ := resp[reqcommon.FieldRequestID].(string)
	if id == "" {
		id = requestID
	}

	choices, _ := resp[reqcommon.FieldChoices].([]any)
	for i, raw := range choices {
		choice, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if _, hasMessage := choice[reqcommon.FieldMessage]; !hasMessage {
			text, _ := choice["text"].(string)
			choice[reqcommon.FieldMessage] = map[string]any{
				reqcommon.FieldRole:    reqcommon.RoleAssistant,
				reqcommon.FieldContent: text,
			}
		}
		delete(choice, "text")
		for _, field := range generateOnlyChoiceFields {
			delete(choice, field)
		}
		choices[i] = choice
	}

	for _, field := range generateOnlyResponseFields {
		delete(resp, field)
	}
	resp[reqcommon.FieldID] = id
	resp[reqcommon.FieldObject] = reqcommon.ObjectChatCompletion
	// choices is required by the schema, so an answer that carried none is
	// still well-formed rather than missing the key.
	if choices == nil {
		choices = []any{}
	}
	resp[reqcommon.FieldChoices] = choices

	wrapped, err := json.Marshal(resp)
	if err != nil {
		return nil, fmt.Errorf("encoding chat completion response: %w", err)
	}
	return wrapped, nil
}
