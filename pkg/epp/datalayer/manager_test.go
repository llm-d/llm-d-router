package datalayer

import (
	"fmt"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	extractormocks "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/mocks"
	srcmocks "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/mocks"
)

// TestConcurrentAccessRaceFree pins concurrent safety on the manager maps.
// Run under -race to catch regressions.
func TestConcurrentAccessRaceFree(t *testing.T) {
	const goroutines = 32

	cases := []struct {
		name string
		run  func(t *testing.T)
	}{
		{
			name: "variantSourceMap reads",
			run: func(t *testing.T) {
				m := newVariantSourceMap[fwkdl.PollingDataSource](variantPolling)
				for i := 0; i < 5; i++ {
					m.Set(srcmocks.NewDataSource(fwkplugin.TypedName{Type: "polling", Name: fmt.Sprintf("src%d", i)}))
				}
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
			},
		},
		{
			name: "extractorMap reads",
			run: func(t *testing.T) {
				m := newExtractorMap()
				for i := 0; i < 5; i++ {
					m.Append(fmt.Sprintf("src%d", i),
						extractormocks.NewEndpointExtractor(fmt.Sprintf("ext%d", i)))
				}
				var wg sync.WaitGroup
				for i := 0; i < goroutines; i++ {
					wg.Add(1)
					go func(i int) {
						defer wg.Done()
						_, _ = m.Get(fmt.Sprintf("src%d", i%5))
						_ = m.Count()
						m.Range(func(string, []fwkdl.ExtractorBase) bool { return true })
					}(i)
				}
				wg.Wait()
			},
		},
		{
			name: "extractorMap concurrent Append dedups",
			run: func(t *testing.T) {
				m := newExtractorMap()
				const srcName = "shared"
				var wg sync.WaitGroup
				for i := 0; i < goroutines; i++ {
					wg.Add(1)
					go func(i int) {
						defer wg.Done()
						m.Append(srcName, extractormocks.NewEndpointExtractor(fmt.Sprintf("ext%d", i)))
					}(i)
				}
				wg.Wait()
				got, _ := m.Get(srcName)
				require.Len(t, got, 1, "Append must dedup by Type under concurrent callers")
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, tc.run)
	}
}
