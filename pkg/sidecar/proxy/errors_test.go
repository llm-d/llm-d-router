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
	"errors"
	"net/http"
	"net/http/httptest"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive
)

var _ = Describe("sendError", func() {
	// Without an explicit header, net/http sniffs the JSON body as text/plain.
	DescribeTable("labels the vLLM error body as JSON",
		func(send func(error, http.ResponseWriter) error, code int) {
			w := httptest.NewRecorder()

			Expect(send(errors.New("boom"), w)).To(Succeed())

			resp := w.Result()
			Expect(resp.StatusCode).To(Equal(code))
			Expect(resp.Header.Get("Content-Type")).To(Equal("application/json"))
		},
		Entry("bad request", errorJSONInvalid, http.StatusBadRequest),
		Entry("bad gateway", errorBadGateway, http.StatusBadGateway),
		Entry("internal server error", errorInternalServerError, http.StatusInternalServerError),
	)
})
