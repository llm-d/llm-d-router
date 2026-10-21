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

	. "github.com/onsi/ginkgo/v2" //nolint:revive
	. "github.com/onsi/gomega"    //nolint:revive

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

var _ = Describe("sendError", func() {
	// Without an explicit header, net/http sniffs the JSON body as text/plain.
	DescribeTable("writes the error envelope of the request's API",
		func(send func(http.ResponseWriter, *http.Request, error) error, path string, code int, want string) {
			w := httptest.NewRecorder()

			Expect(send(w, httptest.NewRequest(http.MethodPost, path, nil), errors.New("boom <&>"))).To(Succeed())

			resp := w.Result()
			Expect(resp.StatusCode).To(Equal(code))
			Expect(resp.Header.Get("Content-Type")).To(Equal("application/json"))
			Expect(w.Body.String()).To(MatchJSON(want))
		},
		Entry("bad request on chat completions", errorJSONInvalid, reqcommon.PathChatCompletions, http.StatusBadRequest,
			`{"error":{"message":"boom <&>","type":"invalid_request_error","code":400}}`),
		Entry("bad gateway on responses", errorBadGateway, reqcommon.PathResponses, http.StatusBadGateway,
			`{"error":{"message":"boom <&>","type":"server_error","code":502}}`),
		Entry("internal server error on completions", errorInternalServerError, reqcommon.PathCompletions, http.StatusInternalServerError,
			`{"error":{"message":"boom <&>","type":"server_error","code":500}}`),
		Entry("bad request on messages", errorJSONInvalid, reqcommon.PathMessages, http.StatusBadRequest,
			`{"type":"error","error":{"type":"invalid_request_error","message":"boom <&>"}}`),
		Entry("bad gateway on messages", errorBadGateway, reqcommon.PathMessages, http.StatusBadGateway,
			`{"type":"error","error":{"type":"api_error","message":"boom <&>"}}`),
		// The decoder catch-all proxies paths that are not inference routes.
		Entry("bad gateway on a path that is not an inference route", errorBadGateway, reqcommon.PathModels, http.StatusBadGateway,
			`{"error":{"message":"boom <&>","type":"server_error","code":502}}`),
	)
})
