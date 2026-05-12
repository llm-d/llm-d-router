package datalayer

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	srcmocks "github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/plugins/datalayer/source/mocks"
)

// TestVariantSourceMap_ConcurrentReadsRaceFree verifies variantSourceMap's
// sync.Map backing permits concurrent read access from many goroutines
// without data races. Run under -race to catch regressions if the storage
// switches to a primitive that requires explicit locking on reads.
func TestVariantSourceMap_ConcurrentReadsRaceFree(t *testing.T) {
	m := newVariantSourceMap[fwkdl.PollingDataSource](variantPolling)
	for i := 0; i < 5; i++ {
		m.Set(srcmocks.NewDataSource(fwkplugin.TypedName{Type: "polling", Name: fmt.Sprintf("src%d", i)}))
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
			m.Range(func(string, fwkdl.PollingDataSource) bool { return true })
			require.NoError(t, m.ForEach(func(string, fwkdl.PollingDataSource) error { return nil }))
			_ = m.findFirst(func(fwkdl.DataSource) bool { return false })
		}(i)
	}
	wg.Wait()
}
