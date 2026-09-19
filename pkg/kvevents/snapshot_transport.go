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

package kvevents

import (
	"context"
	"net"
	"strings"
	"sync"
	"time"

	zmq "github.com/go-zeromq/zmq4"
	"github.com/go-zeromq/zmq4/transport"
)

// zmq4's TCP dial deadline does not cover the ZMTP handshake. Closing the
// underlying connection on cancellation also interrupts a stalled greeting.
type snapshotTransport struct{ transport.Transport }
type snapshotConn struct {
	net.Conn
	stop func() bool
}

var registerSnapshotTransport = sync.OnceValue(func() error {
	return zmq.RegisterTransport("kv-snapshot-tcp", snapshotTransport{transport.New("tcp")})
})

func (snapshotTransport) Dial(ctx context.Context, dialer transport.Dialer, address string) (net.Conn, error) {
	conn, err := dialer.DialContext(ctx, "tcp", address)
	if err != nil {
		return nil, err
	}
	return &snapshotConn{Conn: conn, stop: context.AfterFunc(ctx, func() { _ = conn.Close() })}, nil
}
func (c *snapshotConn) Read(b []byte) (int, error) {
	if err := c.SetReadDeadline(time.Now().Add(snapshotTimeout)); err != nil {
		return 0, err
	}
	return c.Conn.Read(b)
}
func (c *snapshotConn) Write(b []byte) (int, error) {
	if err := c.SetWriteDeadline(time.Now().Add(snapshotTimeout)); err != nil {
		return 0, err
	}
	return c.Conn.Write(b)
}
func (c *snapshotConn) Close() error { c.stop(); return c.Conn.Close() }
func snapshotTCP(endpoint string) string {
	return strings.Replace(endpoint, "tcp://", "kv-snapshot-tcp://", 1)
}
