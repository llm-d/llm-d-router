// snapcount fetches one KV snapshot from a vLLM publisher and counts its contents.
package main

import (
	"context"
	"encoding/binary"
	"fmt"
	"os"
	"sort"
	"time"

	zmq "github.com/go-zeromq/zmq4"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/llm-d/llm-d-router/pkg/kvevents/engineadapter"
)

type scope struct {
	tier  string
	group int
}

func main() {
	endpoint := os.Args[1]
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()
	req := zmq.NewReq(ctx, zmq.WithDialerMaxRetries(0), zmq.WithDialerTimeout(5*time.Second), zmq.WithTimeout(60*time.Second))
	defer req.Close()
	if err := req.Dial(endpoint); err != nil {
		panic(err)
	}
	start := time.Now()
	if err := req.Send(zmq.NewMsg([]byte("snapshot"))); err != nil {
		panic(err)
	}
	msg, err := req.Recv()
	if err != nil {
		panic(err)
	}
	f := msg.Frames
	fmt.Printf("frames=%d cut=%d fetch=%s\n", len(f), int64(binary.BigEndian.Uint64(f[0])), time.Since(start).Round(time.Millisecond))
	if len(os.Args) > 2 {
		// Dump the reply as length-prefixed frames for offline replay.
		out, err := os.Create(os.Args[2])
		if err != nil {
			panic(err)
		}
		for _, fr := range f {
			var n [4]byte
			binary.BigEndian.PutUint32(n[:], uint32(len(fr)))
			out.Write(n[:])
			out.Write(fr)
		}
		out.Close()
	}
	adapter := engineadapter.NewVLLMAdapter()
	bytes, stored, removed, cleared := 0, 0, 0, 0
	storedBy := map[scope]map[uint64]int{}
	removedBy := map[scope]int{}
	tokens := 0
	for _, chunk := range f[2:] {
		bytes += len(chunk)
		_, _, batch, err := adapter.ParseMessage(&kvevents.RawMessage{Topic: "kv@x:5557@zai-org/GLM-5.3", Payload: chunk})
		if err != nil {
			panic(err)
		}
		for _, ev := range batch.Events {
			switch e := ev.(type) {
			case *kvevents.BlockStoredEvent:
				stored++
				g := -1
				if e.GroupIdx != nil {
					g = *e.GroupIdx
				}
				s := scope{e.DeviceTier, g}
				if storedBy[s] == nil {
					storedBy[s] = map[uint64]int{}
				}
				for _, h := range e.BlockHashes {
					storedBy[s][h]++
				}
				tokens += len(e.Tokens)
			case *kvevents.BlockRemovedEvent:
				removed++
				g := -1
				if e.GroupIdx != nil {
					g = *e.GroupIdx
				}
				removedBy[scope{e.DeviceTier, g}] += len(e.BlockHashes)
			case *kvevents.AllBlocksClearedEvent:
				cleared++
			}
		}
	}
	fmt.Printf("chunks=%d bytes=%.1fMiB events stored=%d removed=%d cleared=%d tokens=%d\n", len(f)-2, float64(bytes)/(1<<20), stored, removed, cleared, tokens)
	keys := make([]scope, 0, len(storedBy))
	for s := range storedBy {
		keys = append(keys, s)
	}
	sort.Slice(keys, func(i, j int) bool { return fmt.Sprint(keys[i]) < fmt.Sprint(keys[j]) })
	for _, s := range keys {
		total, dup := 0, 0
		for _, n := range storedBy[s] {
			total += n
			if n > 1 {
				dup++
			}
		}
		fmt.Printf("  tier=%q group=%d unique_hashes=%d stored_refs=%d hashes_stored_more_than_once=%d removed_hashes=%d\n", s.tier, s.group, len(storedBy[s]), total, dup, removedBy[s])
	}
}
