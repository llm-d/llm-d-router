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

package asyncbroker

import (
	"bytes"
	"context"
	"io"
	"net"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/alicebob/miniredis/v2"
	"github.com/go-logr/logr"
	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/require"
)

const quotaTestKey = "quota:userid:team-a"

// dropScriptReplyConn fails the reply read of the first EVALSHA written on it
// with io.EOF after the server has answered, as a connection reset after the
// script ran would. go-redis then retries the EVALSHA.
type dropScriptReplyConn struct {
	net.Conn
	armed *atomic.Bool
	drop  bool
}

func (c *dropScriptReplyConn) Write(b []byte) (int, error) {
	if bytes.Contains(bytes.ToLower(b), []byte("\r\nevalsha\r\n")) && c.armed.CompareAndSwap(true, false) {
		c.drop = true
	}
	return c.Conn.Write(b)
}

func (c *dropScriptReplyConn) Read(b []byte) (int, error) {
	if c.drop {
		_, _ = c.Conn.Read(b) // the reply arrived, so the script ran
		_ = c.Close()
		return 0, io.EOF
	}
	return c.Conn.Read(b)
}

// newLosingReplyClassifier returns a classifier for team-a with the given
// limit. The reply to the next script run after armed is set is lost. The
// scripts are loaded up front so Script.Run executes them through EVALSHA
// instead of answering NOSCRIPT first.
func newLosingReplyClassifier(t *testing.T, limit int) (*asyncQuotaClassifier, *miniredis.Miniredis, *atomic.Bool) {
	t.Helper()
	mr := miniredis.RunT(t)
	armed := &atomic.Bool{}
	rdb := redis.NewClient(&redis.Options{
		Addr: mr.Addr(),
		Dialer: func(ctx context.Context, network, addr string) (net.Conn, error) {
			conn, err := (&net.Dialer{}).DialContext(ctx, network, addr)
			if err != nil {
				return nil, err
			}
			return &dropScriptReplyConn{Conn: conn, armed: armed}, nil
		},
	})
	t.Cleanup(func() { _ = rdb.Close() })
	require.NoError(t, quotaAcquireScript.Load(t.Context(), rdb).Err())
	require.NoError(t, quotaReleaseScript.Load(t.Context(), rdb).Err())
	return &asyncQuotaClassifier{
		rdb: rdb,
		cfg: asyncQuotaConfig{
			Prefix: "quota:", Attribute: "userid",
			Limits: map[string]int{"team-a": limit}, WindowSeconds: 60,
		},
		logger: logr.Discard(),
	}, mr, armed
}

func requireQuotaCount(t *testing.T, mr *miniredis.Miniredis, want string, msg string) {
	t.Helper()
	got, err := mr.Get(quotaTestKey)
	require.NoError(t, err)
	require.Equal(t, want, got, msg)
}

func TestAsyncQuotaClassifyRetryAfterLostReplyTakesOneSlot(t *testing.T) {
	q, mr, armed := newLosingReplyClassifier(t, 5)

	armed.Store(true)
	class, release, err := q.classify(t.Context(), "team-a")
	require.NoError(t, err)
	require.False(t, armed.Load(), "the classify reply was dropped")
	require.Equal(t, asyncClassificationReserved, class)
	require.NotNil(t, release)
	requireQuotaCount(t, mr, "1", "one classify takes one slot")

	release()
	requireQuotaCount(t, mr, "0", "no slot is held after the release")
}

func TestAsyncQuotaClassifyRetryAfterLostReplyAtTheLastSlotIsReserved(t *testing.T) {
	q, mr, armed := newLosingReplyClassifier(t, 1)

	armed.Store(true)
	class, release, err := q.classify(t.Context(), "team-a")
	require.NoError(t, err)
	require.False(t, armed.Load(), "the classify reply was dropped")
	require.Equal(t, asyncClassificationReserved, class, "the first attempt took the slot, so the retry must report it")
	require.NotNil(t, release)
	requireQuotaCount(t, mr, "1", "one classify takes one slot")
}

func TestAsyncQuotaReleaseRetryAfterLostReplyReturnsOneSlot(t *testing.T) {
	q, mr, armed := newLosingReplyClassifier(t, 2)

	_, release1, err := q.classify(t.Context(), "team-a")
	require.NoError(t, err)
	require.NotNil(t, release1)
	_, release2, err := q.classify(t.Context(), "team-a")
	require.NoError(t, err)
	require.NotNil(t, release2)

	armed.Store(true)
	release1()
	require.False(t, armed.Load(), "the release reply was dropped")
	requireQuotaCount(t, mr, "1", "the other holder keeps its slot")
}

func TestAsyncQuotaCallGuardsExpire(t *testing.T) {
	q, mr, _ := newLosingReplyClassifier(t, 2)

	_, release, err := q.classify(t.Context(), "team-a")
	require.NoError(t, err)
	require.NotNil(t, release)
	release()

	guards := 0
	for _, key := range mr.Keys() {
		if key == quotaTestKey {
			continue
		}
		guards++
		require.True(t, strings.HasPrefix(key, "quota-call:"+quotaTestKey+":"), key)
		require.Equal(t, quotaCallGuardTTL, mr.TTL(key), key)
	}
	require.Equal(t, 2, guards, "one guard for the acquire and one for the release")
}
