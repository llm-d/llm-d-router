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
	})
})
