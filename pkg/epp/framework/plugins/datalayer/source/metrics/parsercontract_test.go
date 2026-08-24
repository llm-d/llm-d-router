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
