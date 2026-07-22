// Package utils contains utilities for testing
//
//revive:disable:var-naming
package utils

//revive:enable:var-naming

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
