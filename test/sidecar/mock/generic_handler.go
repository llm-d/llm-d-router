/*
Copyright 2025 The llm-d Authors.

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

package mock

import (
	"io"
	"net/http"
	"sync/atomic"
)

// GenericHandler is a simple mock handler counting incoming requests
type GenericHandler struct {
	RequestCount atomic.Int32
}

func (cc *GenericHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	cc.RequestCount.Add(1)

	defer r.Body.Close() //nolint:errcheck
	_, err := io.ReadAll(r.Body)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest) // TODO: check FastAPI error code when failing to read body
		w.Write([]byte(err.Error()))         //nolint:errcheck
		return
	}

	w.WriteHeader(200)
}
