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

package request

import (
	"reflect"
	"testing"
)

func TestP2PPeerParams(t *testing.T) {
	got := P2PPeerParams("k", "10.0.0.7", 7779)
	want := map[string]any{"kv_request_id": "k", "remote_host": "10.0.0.7", "remote_port": 7779}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("P2PPeerParams() = %v, want %v", got, want)
	}
	if len(got) != 3 {
		t.Errorf("P2PPeerParams() has %d keys, want 3", len(got))
	}
}
