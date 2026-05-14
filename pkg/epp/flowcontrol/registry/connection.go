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

package registry

import (
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/flowcontrol/contracts"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/flowcontrol"
)

// connection is the concrete, un-exported implementation of the contracts.ActiveFlowConnection interface.
// It represents a scoped lease that pins the flow state in memory, preventing garbage collection for the duration of
// the session.
type connection struct {
	registry *FlowRegistry
	key      flowcontrol.FlowKey
}

var _ contracts.ActiveFlowConnection = &connection{}

// GetShard returns the shard this connection is pinned to.
func (c *connection) GetShard() contracts.RegistryShard {
	return c.registry.shard
}

// FlowKey returns the immutable identity of the flow this connection is pinned to.
func (c *connection) FlowKey() flowcontrol.FlowKey {
	return c.key
}
