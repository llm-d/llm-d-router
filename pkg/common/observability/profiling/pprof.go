/*
Copyright 2025 The Kubernetes Authors.

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

package profiling

import (
	"net/http"
	"net/http/pprof"
	"runtime"

	ctrl "sigs.k8s.io/controller-runtime"
)

// PprofHandlers returns the pprof endpoints to mount on a metrics server,
// keyed by URL path. The runtime block/mutex profile rates are turned on as a
// side effect so the corresponding handlers return non-empty data; safe to
// call once per process.
func PprofHandlers() map[string]http.Handler {
	runtime.SetMutexProfileFraction(1)
	runtime.SetBlockProfileRate(1)
	return map[string]http.Handler{
		"/debug/pprof/":             http.HandlerFunc(pprof.Index),
		"/debug/pprof/cmdline":      http.HandlerFunc(pprof.Cmdline),
		"/debug/pprof/profile":      http.HandlerFunc(pprof.Profile),
		"/debug/pprof/symbol":       http.HandlerFunc(pprof.Symbol),
		"/debug/pprof/trace":        http.HandlerFunc(pprof.Trace),
		"/debug/pprof/heap":         pprof.Handler("heap"),
		"/debug/pprof/goroutine":    pprof.Handler("goroutine"),
		"/debug/pprof/allocs":       pprof.Handler("allocs"),
		"/debug/pprof/threadcreate": pprof.Handler("threadcreate"),
		"/debug/pprof/block":        pprof.Handler("block"),
		"/debug/pprof/mutex":        pprof.Handler("mutex"),
	}
}

// SetupPprofHandlers registers the pprof endpoints on the controller-runtime
// manager's metrics server.
func SetupPprofHandlers(mgr ctrl.Manager) error {
	for path, h := range PprofHandlers() {
		if err := mgr.AddMetricsServerExtraHandler(path, h); err != nil {
			return err
		}
	}
	return nil
}
