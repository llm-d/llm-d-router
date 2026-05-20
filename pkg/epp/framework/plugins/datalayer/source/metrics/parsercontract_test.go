package metrics

import (
	"testing"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/http/httptest"
)

func TestParseMetrics_Contract(t *testing.T) {
	httptest.ParserContract(t, parseMetrics,
		[]byte(""),
		[]byte("# HELP foo_total counts foo\n# TYPE foo_total counter\nfoo_total 1\n"),
		[]byte("# HELP queue depth gauge\n# TYPE queue gauge\nqueue 1\n# HELP active running gauge\n# TYPE active gauge\nactive 2\n"),
	)
}
