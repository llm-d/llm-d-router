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

package scheduling

import fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"

// RequestDropProducerName is the producer name of RequestDropDataKey. The key is
// shared by every filter that makes a policy decision, so it carries a fixed
// producer name instead of a plugin instance name.
const RequestDropProducerName = "filter"

// RequestDropDataKey is the request attribute a filter writes when it drains the
// candidate set on purpose. SchedulerProfile.Run reads it and returns the typed
// error it describes instead of the generic "no endpoints" error. A filter that
// writes it must declare it in Produces().
var RequestDropDataKey = fwkplugin.NewDataKey("RequestDropDataKey", RequestDropProducerName)

// RequestDrop describes why a filter rejected a request and how the rejection
// must appear on the wire.
type RequestDrop struct {
	// Code is the canonical error code (see pkg/common/error), for example
	// "ServiceUnavailable" or "ResourceExhausted". It selects the HTTP status.
	Code string
	// Reason is the value of the x-llm-d-request-dropped-reason response header.
	Reason string
	// Msg is the human readable message in the error body.
	Msg string
}
