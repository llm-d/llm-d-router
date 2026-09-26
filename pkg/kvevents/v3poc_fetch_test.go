package kvevents_test

import (
	"context"
	"encoding/binary"
	"os"
	"testing"
	"time"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
)

// TestV3PoCFetch fetches a snapshot from a live recorder over the bootstrap
// socket path and writes the frames with 4-byte big-endian lengths, the
// format readSnapshotFile accepts.
func TestV3PoCFetch(t *testing.T) {
	endpoint := os.Getenv("V3POC_ENDPOINT")
	if endpoint == "" {
		t.Skip("V3POC_ENDPOINT not set")
	}
	start := time.Now()
	frames, err := kvevents.FetchSnapshotForTest(context.Background(), endpoint)
	if err != nil {
		t.Fatal(err)
	}
	elapsed := time.Since(start)
	if len(frames) < 2 || len(frames[0]) != 8 || len(frames[1]) != 16 {
		t.Fatalf("malformed snapshot reply: %d frames", len(frames))
	}
	var buf []byte
	total, largest := 0, 0
	for i, f := range frames {
		buf = binary.BigEndian.AppendUint32(buf, uint32(len(f)))
		buf = append(buf, f...)
		if i >= 2 {
			total += len(f)
			largest = max(largest, len(f))
		}
	}
	if err := os.WriteFile(os.Getenv("V3POC_FETCH_OUT"), buf, 0o644); err != nil {
		t.Fatal(err)
	}
	t.Logf("FETCH cut=%d chunks=%d bytes=%d largest_frame=%d elapsed=%s",
		int64(binary.BigEndian.Uint64(frames[0])), len(frames)-2, total, largest, elapsed)
}
