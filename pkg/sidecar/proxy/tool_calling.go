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
	"fmt"

	"github.com/llm-d/llm-d-router/pkg/common/observability/toolcalling"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// toolCallingRequest is owned by the dispatching goroutine. Comparisons for
// asynchronous prefill must be aggregated before starting its goroutine, so
// reporting does not require waiting for prefill to finish.
type toolCallingRequest struct {
	surface  reqcommon.APIType
	original toolcalling.RequestSnapshot
	statuses []toolcalling.FieldStatus
}

// Capture errors disable observation; callers must preserve routing behavior.
func captureToolCallingRequest(apiType reqcommon.APIType, body []byte) (*toolCallingRequest, error) {
	switch apiType {
	case reqcommon.APITypeChatCompletions, reqcommon.APITypeMessages:
	default:
		return nil, fmt.Errorf("unsupported tool-calling API type %s", apiType)
	}
	original, err := toolcalling.CaptureRequestJSON(apiType, body)
	if err != nil {
		return nil, err
	}
	return &toolCallingRequest{surface: apiType, original: original}, nil
}

// compareDispatched must be called at dispatch, after local routing checks.
// Prepared bodies for requests that will not be sent provide no evidence.
// Returned statuses describe this dispatch; the aggregate deduplicates retries
// and decode chunks by keeping the worst outcome for each observed field.
func (r *toolCallingRequest) compareDispatched(body []byte) ([]toolcalling.FieldStatus, error) {
	if r == nil {
		return nil, nil
	}
	after, err := toolcalling.CaptureRequestJSON(r.surface, body)
	if err != nil {
		return nil, err
	}
	statuses, err := toolcalling.CompareRequests(r.original, after)
	if err != nil {
		return nil, err
	}
	r.mergeStatuses(statuses)
	return statuses, nil
}

// rejectFields is only for explicit rejection of the captured client fields.
// Network failures, cancellation, and generic HTTP errors do not establish it.
func (r *toolCallingRequest) rejectFields(fields ...toolcalling.Field) {
	if r != nil {
		r.mergeStatuses(toolcalling.RejectedFieldStatuses(r.original, fields...))
	}
}

func (r *toolCallingRequest) mergeStatuses(statuses []toolcalling.FieldStatus) {
	for _, result := range statuses {
		if !result.Observed {
			continue
		}
		found := false
		for i := range r.statuses {
			if r.statuses[i].Field == result.Field {
				r.statuses[i].Status = worstToolCallingFieldStatus(r.statuses[i].Status, result.Status)
				found = true
				break
			}
		}
		if !found {
			r.statuses = append(r.statuses, result)
		}
	}
}

func (r *toolCallingRequest) fieldStatuses() []toolcalling.FieldStatus {
	if r == nil {
		return nil
	}
	return append([]toolcalling.FieldStatus(nil), r.statuses...)
}

func worstToolCallingFieldStatus(first, second toolcalling.FieldStatusValue) toolcalling.FieldStatusValue {
	worst := toolcalling.FieldStatusPreserved
	for _, status := range []toolcalling.FieldStatusValue{toolcalling.FieldStatusChanged, toolcalling.FieldStatusDropped, toolcalling.FieldStatusRejected} {
		if first == status || second == status {
			worst = status
		}
	}
	return worst
}
