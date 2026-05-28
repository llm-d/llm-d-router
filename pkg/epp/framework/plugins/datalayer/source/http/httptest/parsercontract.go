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
