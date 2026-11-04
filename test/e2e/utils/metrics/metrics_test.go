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
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestGetMetricsReturnsBoilerplateHTTP200(t *testing.T) {
	const boilerplate = "controller_runtime_active_workers 0\ncertwatcher_read_errors_total 0\n"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(boilerplate))
	}))
	t.Cleanup(srv.Close)

	joined := strings.Join(GetMetrics(srv.URL), "\n")
	if !strings.Contains(joined, "certwatcher_read_errors_total") || strings.Contains(joined, "llm_d_epp_info") {
		t.Fatalf("GetMetrics must return the first HTTP 200 body: %q", joined)
	}
}

func TestCallerRetriesUntilEPPRegistry(t *testing.T) {
	const boilerplate = "controller_runtime_active_workers 0\ncertwatcher_read_errors_total 0\n"
	const ready = "controller_runtime_active_workers 0\nllm_d_epp_info{commit=\"test\"} 1\n"

	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusOK)
		body := boilerplate
		if hits.Add(1) > 1 {
			body = ready
		}
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)

	deadline := time.Now().Add(2 * time.Second)
	var joined string
	for {
		joined = strings.Join(GetMetrics(srv.URL), "\n")
		if strings.Contains(joined, "llm_d_epp_info") && !strings.Contains(joined, "certwatcher_read_errors_total") {
			break
		}
		if time.Now().After(deadline) {
			t.Fatalf("did not see llm_d_epp_info without certwatcher boilerplate: %q", joined)
		}
		time.Sleep(10 * time.Millisecond)
	}
	if hits.Load() < 2 {
		t.Fatalf("assertion succeeded after %d scrapes; caller retry must pass the first 200 OK", hits.Load())
	}
}
