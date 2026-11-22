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

package datalayer

import (
	"time"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const pollStatusDataType = "poll-status"

// PollStatus records the current run of failed polls of one polling source
// against one endpoint. The collector writes it as an endpoint attribute under
// PollStatusDataKey when a poll fails and when the first poll after a failure
// succeeds. An endpoint whose source has never failed carries no PollStatus.
type PollStatus struct {
	// ConsecutiveFailures counts the polls that failed since the last
	// successful poll. Zero once a poll succeeds.
	ConsecutiveFailures int
	// FailingSince is when the first poll of the current failure run
	// completed. Zero while ConsecutiveFailures is zero.
	FailingSince time.Time
}

// Clone implements Cloneable.
func (s *PollStatus) Clone() Cloneable {
	if s == nil {
		return nil
	}
	clone := *s
	return &clone
}

// PollStatusDataKey returns the attribute key under which the collector records
// the PollStatus of the polling source with the given plugin name.
func PollStatusDataKey(sourceName string) fwkplugin.DataKey {
	return fwkplugin.NewDataKey(pollStatusDataType, sourceName)
}
