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

package proxy

import (
	"encoding/json"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive
)

var _ = Describe("decodeRequestBody", func() {
	It("decodes inspected fields and keeps the rest as raw bytes", func() {
		tools := `[{"type":"function","function":{"parameters":{"properties":{"b":{},"a":{}}}}}]`
		parsed, err := decodeRequestBody([]byte(`{"stream":true,"max_tokens":5,"tools":` + tools + `}`))
		Expect(err).ToNot(HaveOccurred())

		Expect(parsed[requestFieldStream]).To(BeTrue())
		Expect(parsed[requestFieldMaxTokens]).To(BeNumerically("==", 5))
		Expect(parsed["tools"]).To(Equal(json.RawMessage(tools)))

		out, err := json.Marshal(parsed)
		Expect(err).ToNot(HaveOccurred())
		Expect(string(out)).To(ContainSubstring(`"tools":` + tools))
	})

	It("rejects non-object bodies", func() {
		_, err := decodeRequestBody([]byte(`[1,2]`))
		Expect(err).To(HaveOccurred())

		_, err = decodeRequestBody([]byte(`null`))
		Expect(err).To(HaveOccurred())
	})

	It("keeps messages raw and decodes them on use", func() {
		messages := `[{"role":"user","content":"Hi"}]`
		parsed, err := decodeRequestBody([]byte(`{"messages":` + messages + `}`))
		Expect(err).ToNot(HaveOccurred())
		Expect(parsed[requestFieldMessages]).To(Equal(json.RawMessage(messages)))

		decoded, err := requestMessages(parsed)
		Expect(err).ToNot(HaveOccurred())
		Expect(decoded).To(HaveLen(1))
	})

	It("treats a null messages field as absent", func() {
		parsed, err := decodeRequestBody([]byte(`{"messages":null}`))
		Expect(err).ToNot(HaveOccurred())

		decoded, err := requestMessages(parsed)
		Expect(err).ToNot(HaveOccurred())
		Expect(decoded).To(BeNil())
	})

	It("reports a messages field that is not an array", func() {
		_, err := requestMessages(map[string]any{requestFieldMessages: json.RawMessage(`{}`)})
		Expect(err).To(HaveOccurred())

		_, err = requestMessages(map[string]any{requestFieldMessages: 5})
		Expect(err).To(HaveOccurred())
	})
})
