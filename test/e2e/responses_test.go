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

// Acceptance tests for the /v1/responses endpoint, derived from the
// openresponses compliance suite (https://github.com/openresponses/openresponses).
//
// Each test maps to an openresponses test template ID, noted in the ginkgo.By
// description.  Tests that require backend capabilities not present in the
// vLLM simulator are skipped with an explicit reason.
package e2e

import (
	"bufio"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	responsesPath        = "/v1/responses"
	responsesCompactPath = "/v1/responses/compact"

	// responseObject is the value of the "object" field in a ResponseResource.
	responseObject = "response"
	// responseStatusCompleted is the terminal status returned by the simulator.
	responseStatusCompleted = "completed"
)

// responsesBody builds a /v1/responses request body with the given JSON input
// and optional extra fields (merged as top-level keys).
func responsesBody(inputJSON string, extras map[string]string) string {
	parts := make([]string, 0, 1+len(extras))
	parts = append(parts, fmt.Sprintf(`{"model":%q,"input":%s`, simModelName, inputJSON))
	for k, v := range extras {
		parts = append(parts, fmt.Sprintf(`%q:%s`, k, v))
	}
	return strings.Join(parts, ",") + "}"
}

// userMessageInput returns a single-user-message input array as a JSON string.
func userMessageInput(text string) string {
	return fmt.Sprintf(`[{"type":"message","role":"user","content":%q}]`, text)
}

// assertResponsesBody parses rawBody and asserts that it looks like a
// completed ResponseResource: object="response", status="completed",
// output non-empty, and output[0].role="assistant".
func assertResponsesBody(rawBody []byte) {
	var resp map[string]any
	gomega.Expect(json.Unmarshal(rawBody, &resp)).ShouldNot(gomega.HaveOccurred())
	gomega.Expect(resp["object"]).Should(gomega.Equal(responseObject))
	gomega.Expect(resp["status"]).Should(gomega.Equal(responseStatusCompleted))

	output, ok := resp["output"].([]any)
	gomega.Expect(ok).Should(gomega.BeTrue(), "output must be an array")
	gomega.Expect(output).ShouldNot(gomega.BeEmpty())

	first, ok := output[0].(map[string]any)
	gomega.Expect(ok).Should(gomega.BeTrue(), "output[0] must be an object")
	gomega.Expect(first["role"]).Should(gomega.Equal("assistant"))
}

// parseSSEEvents splits a raw SSE body into individual data payloads,
// excluding the [DONE] sentinel, and returns them as parsed JSON maps.
func parseSSEEvents(sseBody []byte) []map[string]any {
	var events []map[string]any
	scanner := bufio.NewScanner(strings.NewReader(string(sseBody)))
	for scanner.Scan() {
		line := scanner.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			break
		}
		var event map[string]any
		if err := json.Unmarshal([]byte(data), &event); err == nil {
			events = append(events, event)
		}
	}
	return events
}

// findSSETerminalResponse returns the "response" object from the last
// "response.completed" event in the event slice, or nil if not found.
func findSSETerminalResponse(events []map[string]any) map[string]any {
	for i := len(events) - 1; i >= 0; i-- {
		if events[i]["type"] == "response.completed" {
			if r, ok := events[i]["response"].(map[string]any); ok {
				return r
			}
		}
	}
	return nil
}

var _ = ginkgo.Describe("OpenResponses compliance: /v1/responses", ginkgo.Ordered, testWrapper(func() {
	// Each test group spins up a single decode pod and the EPP in the
	// simplest non-P/D configuration — identical to the "non-PD" group in
	// e2e_test.go. The router proxies /v1/responses straight through to the
	// simulator, which implements the full ResponseResource contract.

	ginkgo.BeforeAll(func() {
		createInferencePool(1)
		createModelServersDecode(1)
		createEndPointPicker(simpleConfig)
	})

	// openresponses id: basic-response
	ginkgo.It("basic-response: returns a completed ResponseResource", func() {
		ginkgo.By("POST /v1/responses with a single user message")
		_, _, body := doPost(responsesPath,
			responsesBody(userMessageInput("Say hello in exactly 3 words."), nil),
			nil)

		ginkgo.By("Verifying ResponseResource shape")
		assertResponsesBody(body)
	})

	// openresponses id: system-prompt
	ginkgo.It("system-prompt: accepts system role in input", func() {
		ginkgo.By("POST /v1/responses with a system + user message")
		input := `[{"type":"message","role":"system","content":"You are a helpful assistant."},` +
			`{"type":"message","role":"user","content":"Say hello."}]`
		_, _, body := doPost(responsesPath, responsesBody(input, nil), nil)

		ginkgo.By("Verifying ResponseResource shape")
		assertResponsesBody(body)
	})

	// openresponses id: multi-turn
	ginkgo.It("multi-turn: accepts conversation history as input", func() {
		ginkgo.By("POST /v1/responses with user + assistant + user turns")
		input := `[` +
			`{"type":"message","role":"user","content":"My name is Alice."},` +
			`{"type":"message","role":"assistant","content":"Hello Alice!"},` +
			`{"type":"message","role":"user","content":"What is my name?"}]`
		_, _, body := doPost(responsesPath, responsesBody(input, nil), nil)

		ginkgo.By("Verifying ResponseResource shape")
		assertResponsesBody(body)
	})

	// openresponses id: image-input
	ginkgo.It("image-input: accepts input_image content in user message", func() {
		ginkgo.By("POST /v1/responses with image_url content block")
		input := fmt.Sprintf(`[{"type":"message","role":"user","content":[`+
			`{"type":"input_text","text":"What do you see?"},`+
			`{"type":"input_image","image_url":%q}`+
			`]}]`, testImageURL)
		_, _, body := doPost(responsesPath, responsesBody(input, nil), nil)

		ginkgo.By("Verifying ResponseResource shape")
		assertResponsesBody(body)
	})

	// openresponses id: tool-calling
	ginkgo.It("tool-calling: emits function_call output when tools are provided", func() {
		ginkgo.By("POST /v1/responses with a function tool definition")
		weatherTool := `{"type":"function","name":"get_weather","description":"Get current weather",` +
			`"parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}`
		body := fmt.Sprintf(`{"model":%q,"input":%s,"tools":[%s]}`,
			simModelName,
			userMessageInput("What's the weather in San Francisco?"),
			weatherTool)
		_, _, respBody := doPost(responsesPath, body, nil)

		ginkgo.By("Verifying output contains a function_call item")
		var resp map[string]any
		gomega.Expect(json.Unmarshal(respBody, &resp)).ShouldNot(gomega.HaveOccurred())
		output, ok := resp["output"].([]any)
		gomega.Expect(ok).Should(gomega.BeTrue())
		gomega.Expect(output).ShouldNot(gomega.BeEmpty())

		hasToolCall := false
		for _, item := range output {
			if m, ok := item.(map[string]any); ok {
				if m["type"] == "function_call" {
					hasToolCall = true
					break
				}
			}
		}
		gomega.Expect(hasToolCall).Should(gomega.BeTrue(), "expected a function_call item in output")
	})

	// openresponses id: compact-missing-model
	ginkgo.It("compact-missing-model: rejects /v1/responses/compact without model field", func() {
		ginkgo.By("POST /v1/responses/compact without model field")
		body := `{"input":[{"type":"message","role":"user","content":"Compact this."}]}`
		status, _ := doPostWithError(responsesCompactPath, body, nil)

		ginkgo.By("Verifying error status 400 or 422")
		gomega.Expect(status).Should(gomega.BeElementOf(400, 422))
	})

	// openresponses id: streaming-response
	ginkgo.It("streaming-response: streams SSE events and terminates with a completed response", func() {
		ginkgo.By("POST /v1/responses with stream:true")
		_, _, sseBody := doPost(responsesPath,
			responsesBody(userMessageInput("Count from 1 to 5."), map[string]string{"stream": "true"}),
			nil)

		ginkgo.By("Parsing SSE event stream")
		events := parseSSEEvents(sseBody)
		gomega.Expect(events).ShouldNot(gomega.BeEmpty(), "expected at least one SSE event before [DONE]")

		ginkgo.By("Verifying a response.created event is present")
		hasCreated := false
		for _, ev := range events {
			if ev["type"] == "response.created" {
				hasCreated = true
				break
			}
		}
		gomega.Expect(hasCreated).Should(gomega.BeTrue(), "expected a response.created event")

		ginkgo.By("Verifying the terminal response.completed event carries a completed ResponseResource")
		terminal := findSSETerminalResponse(events)
		gomega.Expect(terminal).ShouldNot(gomega.BeNil(), "expected a response.completed event")
		gomega.Expect(terminal["object"]).Should(gomega.Equal(responseObject))
		gomega.Expect(terminal["status"]).Should(gomega.Equal(responseStatusCompleted))

		output, ok := terminal["output"].([]any)
		gomega.Expect(ok).Should(gomega.BeTrue())
		gomega.Expect(output).ShouldNot(gomega.BeEmpty())
	})

	// openresponses id: assistant-phase
	// The openresponses spec allows assistant messages to carry a "phase"
	// label (e.g. "commentary", "final_answer"). The simulator accepts
	// these as ordinary assistant turns and produces a completed response.
	ginkgo.It("assistant-phase: accepts assistant messages with phase labels", func() {
		ginkgo.By("POST /v1/responses with phased assistant history")
		input := `[` +
			`{"type":"message","role":"assistant","phase":"commentary","content":"I will answer with the saved number."},` +
			`{"type":"message","role":"assistant","phase":"final_answer","content":"The number is four."},` +
			`{"type":"message","role":"user","content":"Repeat only the number."}]`
		_, _, body := doPost(responsesPath, responsesBody(input, nil), nil)

		ginkgo.By("Verifying ResponseResource shape")
		assertResponsesBody(body)
	})
}))
