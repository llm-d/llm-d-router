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
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime/schema"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	srcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/mocks"
)

// TestVariantSourceMap_ConcurrentReadsRaceFree verifies variantSourceMap's
// sync.Map backing permits concurrent read access from many goroutines
// without data races. Run under -race to catch regressions if the storage
// switches to a primitive that requires explicit locking on reads.
func TestVariantSourceMap_ConcurrentReadsRaceFree(t *testing.T) {
	m := newVariantSourceMap[fwkdl.NotificationSource](variantPolling)
	for i := 0; i < 5; i++ {
		m.Set(srcmocks.NewNotificationSource("polling", fmt.Sprintf("src%d", i), schema.GroupVersionKind{Group: "g", Version: "v", Kind: "k"}))
	}

	const goroutines = 32
	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			_, _ = m.Get(fmt.Sprintf("src%d", i%5))
			_ = m.Sources()
			_ = m.Count()
			_ = m.IsEmpty()
			m.Range(func(string, fwkdl.NotificationSource) bool { return true })
			require.NoError(t, m.ForEach(func(string, fwkdl.NotificationSource) error { return nil }))
			_ = m.findFirst(func(plugin.Plugin) bool { return false })
		}(i)
	}
	wg.Wait()
}
