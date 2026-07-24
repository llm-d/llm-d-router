/*
Copyright 2025 The Kubernetes Authors.

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

package utils

import (
	"fmt"
	"net"
	"testing"
)

func TestGetFreePort(t *testing.T) {
	// Allocating several ports while holding each listener open proves three
	// properties at once: every port is in range, every port is bindable, and no
	// two allocations collide. Distinctness holds only while the prior port stays
	// bound - GetFreePort closes its own listener, so two bare calls may repeat a
	// port; keeping the listener open is what forces the OS to hand out a new one.
	const numPorts = 8

	seen := make(map[int]bool, numPorts)
	held := make([]net.Listener, 0, numPorts)
	t.Cleanup(func() {
		for _, ln := range held {
			_ = ln.Close()
		}
	})

	for i := 0; i < numPorts; i++ {
		port, err := GetFreePort()
		if err != nil {
			t.Fatalf("call %d: GetFreePort() returned error: %v", i, err)
		}
		if port <= 0 || port > 65535 {
			t.Fatalf("call %d: GetFreePort() returned out-of-range port %d", i, port)
		}
		if seen[port] {
			t.Fatalf("call %d: GetFreePort() returned duplicate port %d while it was still bound", i, port)
		}
		seen[port] = true

		// The returned port must be bindable on IPv4 localhost; hold it open so the
		// next iteration cannot be handed the same port back.
		ln, err := net.Listen("tcp", fmt.Sprintf("127.0.0.1:%d", port))
		if err != nil {
			t.Fatalf("call %d: port %d returned by GetFreePort() is not bindable: %v", i, port, err)
		}
		held = append(held, ln)
	}
}
