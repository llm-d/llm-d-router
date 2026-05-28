package models

import (
	"testing"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/http/httptest"
)

func TestParseModels_Contract(t *testing.T) {
	httptest.ParserContract(t, parseModels,
		[]byte(`{"object":"list","data":[]}`),
		[]byte(`{"object":"list","data":[{"id":"qwen2.5-7b"}]}`),
		[]byte(`{"object":"list","data":[{"id":"a","parent":"a"},{"id":"b"}]}`),
	)
}
