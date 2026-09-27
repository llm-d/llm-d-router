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

// Package httptest provides test helpers for HTTPDataSource parser
// implementations.
package httptest

import (
	"bytes"
	"io"
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"
)

// ParserContract asserts that parser satisfies the HTTPDataSource parser
// contract: each goldenInput returns (meaningful T, nil error), and never
// (nil, nil) for nilable T.
func ParserContract[T any](t *testing.T, parser func(io.Reader) (T, error), goldenInputs ...[]byte) {
	t.Helper()
	nilable := isNilable(reflect.TypeFor[T]().Kind())
	for i, input := range goldenInputs {
		value, err := parser(bytes.NewReader(input))
		require.NoErrorf(t, err, "golden input %d: parser returned error", i)
		if nilable {
			require.Falsef(t, reflect.ValueOf(value).IsNil(),
				"golden input %d: parser returned nil for nilable T (contract violation)", i)
		}
	}
}

func isNilable(k reflect.Kind) bool {
	switch k {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Chan, reflect.Func, reflect.Interface:
		return true
	}
	return false
}
