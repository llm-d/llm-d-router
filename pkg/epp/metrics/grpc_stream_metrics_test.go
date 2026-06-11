package metrics

import (
	"testing"

	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
)

func TestExtProcStreamMetrics(t *testing.T) {
	extProcStreamsInflight.Set(0)
	extProcStreamsTotal.Reset()

	require.Equal(t, 0.0, promtestutil.ToFloat64(extProcStreamsInflight))

	// inflight tracks open streams; close decrements and counts by code.
	ExtProcStreamStarted()
	ExtProcStreamStarted()
	require.Equal(t, 2.0, promtestutil.ToFloat64(extProcStreamsInflight))

	ExtProcStreamFinished("OK", 0.42)
	require.Equal(t, 1.0, promtestutil.ToFloat64(extProcStreamsInflight))
	require.Equal(t, 1.0, promtestutil.ToFloat64(extProcStreamsTotal.WithLabelValues("OK")))

	ExtProcStreamFinished("Internal", 0.1)
	require.Equal(t, 0.0, promtestutil.ToFloat64(extProcStreamsInflight))
	require.Equal(t, 1.0, promtestutil.ToFloat64(extProcStreamsTotal.WithLabelValues("Internal")))
	require.Equal(t, 1, promtestutil.CollectAndCount(extProcStreamDuration))
}
