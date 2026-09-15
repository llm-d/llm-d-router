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

package sessionprefixcache

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"maps"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
)

var _ requestcontrol.PreRequest = &Producer{}

// PreRequest writes the request's SessionID into the body's session_id field,
// replacing any client value, and sets vllm_xargs.kv_cache_report_mode to
// full or incremental. vLLM reads the body field ahead of the session header
// and vllm_xargs.session_id, so the engine reports this request's blocks
// under the session producer's identity. Other body fields are preserved.
// No-op for a request without a session identity.
func (p *Producer) PreRequest(_ context.Context, request *scheduling.InferenceRequest, _ *scheduling.SchedulingResult) error {
	if request == nil {
		return nil
	}
	lookup, ok := scheduling.ReadRequestAttribute[attrsession.SessionCacheRequest](request, p.sessionDK)
	if !ok || lookup.SessionID == "" {
		return nil
	}
	if request.Body == nil {
		return errors.New("session stamping requires a JSON request body")
	}
	payload, ok := request.Body.Payload.(fwkrh.PayloadMap)
	if !ok {
		return errors.New("session stamping requires a JSON request envelope")
	}
	xargs, err := extraArgs(payload["vllm_xargs"])
	if err != nil {
		return err
	}
	mode := "incremental"
	if lookup.FullReport {
		mode = "full"
	}
	xargs["kv_cache_report_mode"] = mode
	request.Body.MutatePayloadMap(func(payload fwkrh.PayloadMap) {
		payload["session_id"] = lookup.SessionID
		payload["vllm_xargs"] = xargs
	})
	return nil
}

// extraArgs returns a copy of the request's vllm_xargs object. A raw message
// is decoded with number precision preserved.
func extraArgs(raw any) (map[string]any, error) {
	var xargs map[string]any
	switch existing := raw.(type) {
	case nil:
	case map[string]any:
		xargs = maps.Clone(existing)
	case fwkrh.PayloadMap:
		xargs = maps.Clone(map[string]any(existing))
	case json.RawMessage:
		decoder := json.NewDecoder(bytes.NewReader(existing))
		decoder.UseNumber()
		if err := decoder.Decode(&xargs); err != nil || xargs == nil {
			return nil, errors.New("session stamping requires an object for vllm_xargs")
		}
	default:
		return nil, errors.New("session stamping requires an object for vllm_xargs")
	}
	if xargs == nil {
		xargs = map[string]any{}
	}
	return xargs, nil
}
