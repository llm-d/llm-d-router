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
	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive
	"github.com/tidwall/gjson"
)

var _ = Describe("validateRequestBody", func() {
	It("accepts a JSON object", func() {
		Expect(validateRequestBody([]byte(`{"a":1}`))).To(Succeed())
	})

	It("rejects invalid JSON and non-object bodies", func() {
		Expect(validateRequestBody([]byte(`{"a":`))).ToNot(Succeed())
		Expect(validateRequestBody([]byte(`[1,2]`))).ToNot(Succeed())
	})
})

var _ = Describe("requestBody", func() {
	tools := `[{"type":"function","function":{"parameters":{"properties":{"b":{},"a":{}}}}}]`
	original := `{"stream":true,"stream_options":{"include_usage":true},"max_tokens":5,"tools":` + tools + `}`

	It("rewrites only the edited fields and keeps the rest byte-for-byte", func() {
		out, err := editRequestBody([]byte(original)).
			set(requestFieldStream, false).
			del(requestFieldStreamOptions).
			set(requestFieldSamplingParams+"."+requestFieldMaxTokens, 1).
			bytes()
		Expect(err).ToNot(HaveOccurred())

		Expect(string(out)).To(ContainSubstring(`"tools":` + tools))
		Expect(gjson.GetBytes(out, requestFieldStream).Bool()).To(BeFalse())
		Expect(gjson.GetBytes(out, requestFieldStreamOptions).Exists()).To(BeFalse())
		Expect(gjson.GetBytes(out, requestFieldMaxTokens).Int()).To(BeEquivalentTo(5))
		Expect(gjson.GetBytes(out, requestFieldSamplingParams+"."+requestFieldMaxTokens).Int()).To(BeEquivalentTo(1))
	})

	It("does not modify the input body", func() {
		in := []byte(original)
		_, err := editRequestBody(in).set(requestFieldStream, false).bytes()
		Expect(err).ToNot(HaveOccurred())
		Expect(string(in)).To(Equal(original))
	})

	It("singleToken caps max_completion_tokens only when present", func() {
		out, err := editRequestBody([]byte(original)).singleToken().bytes()
		Expect(err).ToNot(HaveOccurred())
		Expect(gjson.GetBytes(out, requestFieldMaxTokens).Int()).To(BeEquivalentTo(1))
		Expect(gjson.GetBytes(out, requestFieldMaxCompletionTokens).Exists()).To(BeFalse())
		Expect(gjson.GetBytes(out, requestFieldStream).Bool()).To(BeFalse())
		Expect(gjson.GetBytes(out, requestFieldStreamOptions).Exists()).To(BeFalse())

		out, err = editRequestBody([]byte(`{"max_completion_tokens":100}`)).singleToken().bytes()
		Expect(err).ToNot(HaveOccurred())
		Expect(gjson.GetBytes(out, requestFieldMaxCompletionTokens).Int()).To(BeEquivalentTo(1))
	})
})
