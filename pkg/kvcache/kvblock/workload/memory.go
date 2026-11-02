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

package workload

import (
	"os"
	"runtime"
	"strconv"
	"strings"
	"time"
)

// HeapSnapshot samples live-heap bytes and GC counters at one point in
// time, after forcing a collection so HeapAlloc reports live objects only.
type HeapSnapshot struct {
	HeapAlloc  uint64
	PauseTotal time.Duration
	NumGC      uint32
}

// ReadHeap forces a GC, then samples live heap bytes and cumulative GC
// counters. Differences between two snapshots give a region's steady-state
// cost and its GC overhead.
func ReadHeap() HeapSnapshot {
	runtime.GC()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	return HeapSnapshot{
		HeapAlloc:  ms.HeapAlloc,
		PauseTotal: time.Duration(ms.PauseTotalNs), //#nosec G115 -- nanoseconds of pause, far inside int64
		NumGC:      ms.NumGC,
	}
}

// Delta returns after-before, zero when before exceeds after: a forced
// collection or OS page reclamation can drop a later snapshot below an
// earlier one, and callers report the difference as unsigned bytes.
func Delta(after, before uint64) uint64 {
	if after > before {
		return after - before
	}
	return 0
}

// RSSBytes returns the process's resident set size, the figure the
// container limit and OOM killer enforce. Zero when /proc is unavailable.
func RSSBytes() uint64 {
	data, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		return 0
	}
	fields := strings.Fields(string(data))
	if len(fields) < 2 {
		return 0
	}
	pages, err := strconv.ParseUint(fields[1], 10, 64)
	if err != nil {
		return 0
	}
	return pages * uint64(os.Getpagesize()) //#nosec G115 -- Getpagesize is positive
}
