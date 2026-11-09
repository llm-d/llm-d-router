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

package internal

import (
	"testing"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkfcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol/mocks"
)

func TestChooseDispatchItemSkipsHeldHead(t *testing.T) {
	t.Parallel()

	miss := &fwkfcmocks.MockQueueItemAccessor{}
	hit := &fwkfcmocks.MockQueueItemAccessor{}
	band := &fwkfcmocks.MockPriorityBandAccessor{
		IterateQueuesFunc: func(callback func(flow flowcontrol.FlowQueueAccessor) bool) {
			callback(&fwkfcmocks.MockFlowQueueAccessor{PeekV: miss})
			callback(&fwkfcmocks.MockFlowQueueAccessor{PeekV: hit})
		},
	}
	allowHit := func(item flowcontrol.QueueItemAccessor) bool { return item == hit }

	if got := chooseDispatchItem(miss, band, allowHit); got != hit {
		t.Fatalf("got %v, want the sibling hit", got)
	}
	if got := chooseDispatchItem(hit, band, allowHit); got != hit {
		t.Fatalf("an allowed pick should win without a scan, got %v", got)
	}
	allowNone := func(flowcontrol.QueueItemAccessor) bool { return false }
	if got := chooseDispatchItem(miss, band, allowNone); got != nil {
		t.Fatalf("got %v, want nil when every head is held", got)
	}
}
