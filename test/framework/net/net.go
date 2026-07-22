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

// Package net provides network helpers for tests.
package net

import (
	"errors"
	"fmt"
	"net"
)

// GetFreePort finds an available IPv4 TCP port on localhost.
// It works by asking the OS to allocate a port by listening on port 0, capturing the assigned address, and then
// immediately closing the listener.
//
// Note: There is a theoretical race condition where another process grabs the port between the Close() call and the
// subsequent usage, but this is generally acceptable in hermetic test environments.
func GetFreePort() (int, error) {
	// Force IPv4 to prevent flakes on dual-stack CI environments
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return 0, fmt.Errorf("failed to listen on a free port: %w", err)
	}

	// Critical: Close the listener immediately so the caller can bind to it.
	defer listener.Close()

	addr, ok := listener.Addr().(*net.TCPAddr)
	if !ok {
		return 0, errors.New("failed to cast listener address to TCPAddr")
	}
	return addr.Port, nil
}
