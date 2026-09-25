package kvevents_test

import (
	"bufio"
	"context"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"strconv"
	"testing"

	"github.com/vmihailenco/msgpack/v5"

	"github.com/llm-d/llm-d-router/pkg/kvcache/kvblock"
	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
)

// readSnapshotFile returns the cut sequence and reply chunks of a captured
// snapshot: either a msgpack array [seq, stream_id, chunk...] or the ZMQ
// frames written back to back with 4-byte big-endian lengths.
func readSnapshotFile(t *testing.T, path string) (int64, [][]byte) {
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) > 4 && binary.BigEndian.Uint32(raw[:4]) == 8 {
		var frames [][]byte
		for i := 0; i+4 <= len(raw); {
			n := int(binary.BigEndian.Uint32(raw[i : i+4]))
			frames = append(frames, raw[i+4:i+4+n])
			i += 4 + n
		}
		return int64(binary.BigEndian.Uint64(frames[0])), frames[2:]
	}
	var parts []any
	if err := msgpack.Unmarshal(raw, &parts); err != nil {
		t.Fatal(err)
	}
	seq, err := strconv.ParseInt(fmt.Sprint(parts[0]), 10, 64)
	if err != nil {
		t.Fatal(err)
	}
	chunks := make([][]byte, 0, len(parts)-2)
	for _, p := range parts[2:] {
		chunks = append(chunks, p.([]byte))
	}
	return seq, chunks
}

type liveFrame struct {
	seq     int64
	payload []byte
}

func readLiveFile(t *testing.T, path string) []liveFrame {
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	r := bufio.NewReader(f)
	var out []liveFrame
	for {
		var n uint32
		if err := binary.Read(r, binary.BigEndian, &n); err != nil {
			if err == io.EOF || err == io.ErrUnexpectedEOF {
				return out
			}
			t.Fatal(err)
		}
		buf := make([]byte, n)
		if _, err := io.ReadFull(r, buf); err != nil {
			return out
		}
		var rec []any
		if err := msgpack.Unmarshal(buf, &rec); err != nil {
			t.Fatal(err)
		}
		seq, err := strconv.ParseInt(fmt.Sprint(rec[0]), 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		out = append(out, liveFrame{seq: seq, payload: rec[2].([]byte)})
	}
}

// TestV3PoCReplay loads a captured or synthesized snapshot into a strict
// generation pool, follows live frames after the cut, and writes the owned
// index entries and dedup reference counts for comparison.
func TestV3PoCReplay(t *testing.T) {
	snapPath := os.Getenv("V3POC_SNAP")
	if snapPath == "" {
		t.Skip("V3POC_SNAP not set")
	}
	until := int64(1 << 62)
	if v := os.Getenv("V3POC_UNTIL"); v != "" {
		var err error
		if until, err = strconv.ParseInt(v, 10, 64); err != nil {
			t.Fatal(err)
		}
	}
	ctx := context.Background()
	tokens, err := kvblock.NewChunkedTokenDatabase(&kvblock.TokenProcessorConfig{BlockSizeTokens: 64})
	if err != nil {
		t.Fatal(err)
	}
	adapter := engineadapter.NewVLLMAdapter()
	adapter.SnapshotMode = true
	index, err := kvblock.NewInMemoryIndex(&kvblock.InMemoryIndexConfig{Size: 10_000_000, PodCacheSize: 1 << 20})
	if err != nil {
		t.Fatal(err)
	}
	cut, chunks := readSnapshotFile(t, snapPath)
	var live [][]byte
	next := cut + 1
	if livePath := os.Getenv("V3POC_LIVE"); livePath != "" {
		for _, frame := range readLiveFile(t, livePath) {
			if frame.seq < next || frame.seq > until {
				continue
			}
			if frame.seq != next {
				t.Fatalf("live sequence gap: got %d, want %d", frame.seq, next)
			}
			live = append(live, frame.payload)
			next++
		}
	}
	lines, loaded, applied, liveErr, err := kvevents.FollowSnapshotForTest(ctx, index, tokens, adapter,
		"kv@10.0.0.1:8000@zai-org/GLM-5.3", "pod#snapshot-1", chunks, live)
	if err != nil {
		t.Fatal(err)
	}
	if out := os.Getenv("V3POC_OUT"); out != "" {
		f, err := os.Create(out)
		if err != nil {
			t.Fatal(err)
		}
		w := bufio.NewWriter(f)
		for _, line := range lines {
			fmt.Fprintln(w, line)
		}
		if err := w.Flush(); err != nil {
			t.Fatal(err)
		}
		f.Close()
	}
	status := "ok"
	if liveErr != nil {
		status = fmt.Sprintf("FAILED at seq %d: %v", cut+1+int64(applied), liveErr)
	}
	t.Logf("RESULT cut=%d chunks=%d loaded_entries=%d live_applied=%d last_seq=%d lines=%d status=%s",
		cut, len(chunks), loaded, applied, cut+int64(applied), len(lines), status)
	if liveErr != nil {
		t.Fatalf("live replay failed: %v", liveErr)
	}
	if os.Getenv("V3POC_UNTIL") != "" && cut+int64(applied) != until {
		t.Fatalf("live replay stopped at %d, want %d", cut+int64(applied), until)
	}
}
