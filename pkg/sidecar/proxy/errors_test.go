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
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive
	"github.com/stretchr/testify/require"
)

func TestJSONErrorForLog(t *testing.T) {
	require.Nil(t, jsonErrorForLog(nil))
	other := errors.New("read failure")
	require.Same(t, other, jsonErrorForLog(other))
	var target struct {
		Value int `json:"private_schema_property"`
	}
	err := json.Unmarshal([]byte(`{"private_schema_property":987654321098765432109876543210e999}`), &target)
	require.Error(t, err)
	var typeError *json.UnmarshalTypeError
	require.ErrorAs(t, err, &typeError)
	require.Contains(t, err.Error(), "private_schema_property")
	wrapped := fmt.Errorf("private_wrapper: %w", err)
	require.EqualError(t, jsonErrorForLog(wrapped), fmt.Sprintf("invalid JSON value at byte %d", typeError.Offset))
	require.Contains(t, wrapped.Error(), "987654321098765432109876543210e999", "logging must not mutate the original error")
	err = json.Unmarshal([]byte(`{"private_schema_property":`), &target)
	var syntaxError *json.SyntaxError
	require.ErrorAs(t, err, &syntaxError)
	require.EqualError(t, jsonErrorForLog(fmt.Errorf("private_wrapper: %w", err)), fmt.Sprintf("invalid JSON syntax at byte %d", syntaxError.Offset))
}

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
