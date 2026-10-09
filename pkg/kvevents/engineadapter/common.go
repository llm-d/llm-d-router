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

package engineadapter

import "strings"

const (
	// Common event type tags shared across engine adapters.
	eventTagBlockStored      = "BlockStored"
	eventTagBlockRemoved     = "BlockRemoved"
	eventTagAllBlocksCleared = "AllBlocksCleared"
)

// parseTopic extracts pod ID and model name from the topic format "kv@<pod-id>@<model-name>".
//
//nolint:gocritic // unnamedResult: named returns conflict with nonamedreturns linter
func parseTopic(topic string) (string, string) {
	topicParts := strings.Split(topic, "@")
	if len(topicParts) == 3 {
		return topicParts[1], topicParts[2]
	}
	return topic, ""
}
