/*
Copyright 2025 The llm-d Authors.

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

package types

import (
	"encoding/json"
	"errors"
	"strings"
)

// ImageBlock represents the image_url field in a multimodal content block.
type ImageBlock struct {
	URL string `json:"url,omitempty"`
}

// ContentBlock represents a single part of a multimodal message.
type ContentBlock struct {
	Type     string     `json:"type"`
	Text     string     `json:"text,omitempty"`
	ImageURL ImageBlock `json:"image_url,omitempty"`
}

// Content holds a message's content — either plain text or a list of multimodal blocks.
type Content struct {
	Raw        string
	Structured []ContentBlock
}

// UnmarshalJSON handles both the plain-string and block-list formats.
func (c *Content) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err == nil {
		c.Raw = s
		return nil
	}
	var blocks []ContentBlock
	if err := json.Unmarshal(data, &blocks); err == nil {
		c.Structured = blocks
		return nil
	}
	return errors.New("content must be a string or array of content blocks")
}

// MarshalJSON serialises back to the original format.
func (c Content) MarshalJSON() ([]byte, error) {
	if c.Raw != "" {
		return json.Marshal(c.Raw)
	}
	if len(c.Structured) > 0 {
		return json.Marshal(c.Structured)
	}
	return json.Marshal("")
}

// PlainText returns the plain text content, concatenating text blocks for multimodal messages.
func (c Content) PlainText() string {
	if c.Raw != "" {
		return c.Raw
	}
	var sb strings.Builder
	for _, block := range c.Structured {
		if block.Type == "text" {
			sb.WriteString(block.Text)
			sb.WriteString(" ")
		}
	}
	return sb.String()
}

// Conversation represents a single message in a conversation.
type Conversation struct {
	Role      string        `json:"role"`
	Content   Content       `json:"content"`
	ToolCalls []interface{} `json:"tool_calls,omitempty"`
}

// RenderChatRequest represents the request to render a chat template.
type RenderChatRequest struct {
	// The Python wrapper will handle converting this to a batched list if needed.
	Key                       string                 `json:"key"`
	Conversation              []Conversation         `json:"conversation"`
	Tools                     []interface{}          `json:"tools,omitempty"`
	Documents                 []interface{}          `json:"documents,omitempty"`
	ChatTemplate              string                 `json:"chat_template,omitempty"`
	ReturnAssistantTokensMask bool                   `json:"return_assistant_tokens_mask,omitempty"`
	ContinueFinalMessage      bool                   `json:"continue_final_message,omitempty"`
	AddGenerationPrompt       bool                   `json:"add_generation_prompt,omitempty"`
	TruncatePromptTokens      *int                   `json:"truncate_prompt_tokens,omitempty"`
	ChatTemplateKWArgs        map[string]interface{} `json:"chat_template_kwargs,omitempty"`
}

type RenderRequest struct {
	Key              string `json:"key"`
	Text             string `json:"text"`
	AddSpecialTokens bool   `json:"add_special_tokens,omitempty"`
}

// Offset represents a character offset range with [start, end] indices.
type Offset [2]uint

type RenderResponse struct {
	TokenIDs       []uint32 `json:"input_ids"`
	OffsetMappings []Offset `json:"offset_mapping"`
}

// DeepCopy creates a deep copy of the RenderChatRequest.
func (req *RenderChatRequest) DeepCopy() (*RenderChatRequest, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	var out RenderChatRequest
	err = json.Unmarshal(b, &out)
	if err != nil {
		return nil, err
	}
	return &out, nil
}

// DeepCopy creates a deep copy of the RenderRequest.
func (req *RenderRequest) DeepCopy() (*RenderRequest, error) {
	b, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}
	var out RenderRequest
	err = json.Unmarshal(b, &out)
	if err != nil {
		return nil, err
	}
	return &out, nil
}
