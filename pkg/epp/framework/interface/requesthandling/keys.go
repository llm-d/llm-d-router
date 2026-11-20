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

package requesthandling

import "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"

// The keys below let a RequestHeaderProcessor change the request priority and
// tenant that flow control uses. A processor declares the key in Produces() and
// writes it with PutAttribute. After each processor the Director copies the
// value onto the request, so later processors and flow control see it. Writing
// the request fields directly has no effect: each processor works on a copy.
// The keys carry no producer name, so any plugin may produce them.
var (
	// PriorityKey holds the int priority that replaces Objectives.Priority.
	PriorityKey = plugin.NewDataKey("request-priority", "")
	// TenantKey holds the string tenant used as the FairnessID of a request
	// that arrived without a fairness-ID header. The header takes precedence.
	TenantKey = plugin.NewDataKey("request-tenant", "")
)
