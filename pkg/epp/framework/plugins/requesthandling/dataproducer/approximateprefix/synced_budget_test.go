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

package approximateprefix

import "testing"

func TestSyncedHashBudget(t *testing.T) {
	t.Parallel()

	if got := syncedHashBudget(defaultConfig); got != defaultLRUCapacityPerServer {
		t.Fatalf("defaultConfig budget = %d, want %d (index-aligned)", got, defaultLRUCapacityPerServer)
	}
	if defaultMaxSyncedTokens != defaultLRUCapacityPerServer*defaultBlockSizeTokens {
		t.Fatalf("defaultMaxSyncedTokens = %d, want %d", defaultMaxSyncedTokens, defaultLRUCapacityPerServer*defaultBlockSizeTokens)
	}

	cfg := config{BlockSizeTokens: 64, MaxSyncedTokens: 640}
	if got := syncedHashBudget(cfg); got != 10 {
		t.Fatalf("explicit tokens budget = %d, want 10", got)
	}

	cfg = config{BlockSizeTokens: 64, MaxSyncedTokens: 0}
	if got := syncedHashBudget(cfg); got != defaultLRUCapacityPerServer {
		t.Fatalf("unset tokens budget = %d, want %d", got, defaultLRUCapacityPerServer)
	}
}
