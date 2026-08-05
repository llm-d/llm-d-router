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

package proxy

import (
	"context"
	"net"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"
)

func TestHostResolver_RawIPPassthrough(t *testing.T) {
	ctx := context.Background()
	r := newHostResolver(logr.Discard(), time.Minute)
	// Raw IPs must pass through unchanged and never be cached (no lookup).
	require.Equal(t, "10.0.0.1", r.resolveOne(ctx, "10.0.0.1"))
	require.Equal(t, "::1", r.resolveOne(ctx, "::1"))
	require.Empty(t, r.cache)

	require.Equal(t, []string{"10.0.0.1", "10.0.0.2"},
		r.resolve(ctx, []string{"10.0.0.1", "10.0.0.2"}))
}

func TestHostResolver_EmptyPassthrough(t *testing.T) {
	ctx := context.Background()
	r := newHostResolver(logr.Discard(), time.Minute)
	require.Nil(t, r.resolve(ctx, nil))
	require.Empty(t, r.resolve(ctx, []string{}))
}

func TestHostResolver_CachesWithinTTL(t *testing.T) {
	ctx := context.Background()
	r := newHostResolver(logr.Discard(), time.Minute)
	// localhost resolves to a loopback address on all supported platforms.
	first := r.resolveOne(ctx, "localhost")
	require.True(t, first == "127.0.0.1" || first == "::1", "unexpected localhost IP %q", first)
	require.Len(t, r.cache, 1)

	entry := r.cache["localhost"]
	// A second call within the TTL must reuse the cached entry (same resolved time).
	second := r.resolveOne(ctx, "localhost")
	require.Equal(t, first, second)
	require.Equal(t, entry.resolved, r.cache["localhost"].resolved)
}

func TestHostResolver_HardFailurePassesThroughSpec(t *testing.T) {
	r := newHostResolver(logr.Discard(), time.Minute)
	// With no cached value, a hard resolution failure returns the original spec
	// so the downstream engine surfaces the dial error.
	const bad = "unresolvable-host-xyz.invalid"
	require.Equal(t, bad, r.resolveOne(context.Background(), bad))
}

func TestResolveTTLFromEnv(t *testing.T) {
	t.Setenv(envMoRIIODNSResolveTTLSeconds, "45")
	require.Equal(t, 45*time.Second, resolveTTLFromEnv())

	t.Setenv(envMoRIIODNSResolveTTLSeconds, "0")
	require.Equal(t, defaultResolveTTL, resolveTTLFromEnv())

	t.Setenv(envMoRIIODNSResolveTTLSeconds, "notanumber")
	require.Equal(t, defaultResolveTTL, resolveTTLFromEnv())
}

func TestResolveTimeoutFromEnv(t *testing.T) {
	t.Setenv(envMoRIIODNSResolveTimeoutSeconds, "12")
	require.Equal(t, 12*time.Second, resolveTimeoutFromEnv())

	t.Setenv(envMoRIIODNSResolveTimeoutSeconds, "0")
	require.Equal(t, defaultResolveTimeout, resolveTimeoutFromEnv())

	t.Setenv(envMoRIIODNSResolveTimeoutSeconds, "-3")
	require.Equal(t, defaultResolveTimeout, resolveTimeoutFromEnv())

	t.Setenv(envMoRIIODNSResolveTimeoutSeconds, "notanumber")
	require.Equal(t, defaultResolveTimeout, resolveTimeoutFromEnv())
}

// TestHostResolver_SingleflightDedupe verifies that a cold-start stampede of
// concurrent resolveOne calls for the same spec collapses into a single
// underlying DNS lookup, with all callers sharing the result.
func TestHostResolver_SingleflightDedupe(t *testing.T) {
	ctx := context.Background()
	r := newHostResolver(logr.Discard(), time.Minute)

	var calls int32
	release := make(chan struct{})
	// Inject a counting lookup that blocks until released, guaranteeing all
	// goroutines pile up on the same in-flight request before it returns.
	r.lookupIP = func(_ context.Context, host string) ([]net.IPAddr, error) {
		atomic.AddInt32(&calls, 1)
		<-release
		return []net.IPAddr{{IP: net.ParseIP("10.9.8.7")}}, nil
	}

	const n = 25
	var wg sync.WaitGroup
	results := make([]string, n)
	started := make(chan struct{}, n)
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			started <- struct{}{}
			results[idx] = r.resolveOne(ctx, "peer.example.svc")
		}(i)
	}
	// Wait until all goroutines have entered before releasing the lookup.
	for i := 0; i < n; i++ {
		<-started
	}
	// Give the racers a moment to converge on the singleflight group.
	time.Sleep(50 * time.Millisecond)
	close(release)
	wg.Wait()

	require.Equal(t, int32(1), atomic.LoadInt32(&calls),
		"expected exactly one underlying DNS lookup for concurrent callers")
	for _, got := range results {
		require.Equal(t, "10.9.8.7", got)
	}
	require.Equal(t, "10.9.8.7", r.cache["peer.example.svc"].ip)
}

// TestHostResolver_IPv4Preference verifies the injected-lookup path keeps the
// IPv4-preference semantics (prefer an A record over a AAAA record).
func TestHostResolver_IPv4Preference(t *testing.T) {
	r := newHostResolver(logr.Discard(), time.Minute)
	r.lookupIP = func(_ context.Context, _ string) ([]net.IPAddr, error) {
		return []net.IPAddr{
			{IP: net.ParseIP("fd00::1")},     // IPv6 first
			{IP: net.ParseIP("192.168.5.6")}, // IPv4 should win
		}, nil
	}
	require.Equal(t, "192.168.5.6", r.resolveOne(context.Background(), "dual.example.svc"))
}

// TestHostResolver_Timeout verifies a slow cold-start lookup is bounded by the
// resolver timeout and falls back to the original spec when no cache entry
// exists yet.
func TestHostResolver_Timeout(t *testing.T) {
	r := newHostResolver(logr.Discard(), time.Minute)
	r.timeout = 20 * time.Millisecond
	r.lookupIP = func(ctx context.Context, _ string) ([]net.IPAddr, error) {
		<-ctx.Done() // block until the context deadline fires
		return nil, ctx.Err()
	}
	require.Equal(t, "slow.example.svc", r.resolveOne(context.Background(), "slow.example.svc"))
}

// TestHostResolver_ContextCancellationBounds verifies the caller's context
// bounds a cold-start lookup: a cancelled request context cancels the lookup
// even when the resolver's own timeout is long.
func TestHostResolver_ContextCancellationBounds(t *testing.T) {
	r := newHostResolver(logr.Discard(), time.Minute)
	r.timeout = time.Hour // long resolver timeout; ctx must win
	r.lookupIP = func(ctx context.Context, _ string) ([]net.IPAddr, error) {
		<-ctx.Done()
		return nil, ctx.Err()
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // already cancelled
	require.Equal(t, "cancelled.example.svc", r.resolveOne(ctx, "cancelled.example.svc"))
}

// TestHostResolver_ServeStaleWhileRevalidate verifies that once a spec has been
// resolved, a past-TTL entry is served immediately (stale) while an
// asynchronous background refresh updates the cache to the new IP.
func TestHostResolver_ServeStaleWhileRevalidate(t *testing.T) {
	r := newHostResolver(logr.Discard(), 20*time.Millisecond)

	var mu sync.Mutex
	current := "1.1.1.1"
	done := make(chan struct{}, 8)
	r.lookupIP = func(_ context.Context, _ string) ([]net.IPAddr, error) {
		mu.Lock()
		ip := current
		mu.Unlock()
		res := []net.IPAddr{{IP: net.ParseIP(ip)}}
		done <- struct{}{} // signal the lookup ran (before doLookup writes cache)
		return res, nil
	}

	// Cold start: blocks, seeds the cache with 1.1.1.1.
	require.Equal(t, "1.1.1.1", r.resolveOne(context.Background(), "peer"))
	<-done // consume the cold-start lookup signal

	// Point the backing name at a new IP and let the cached entry go stale.
	mu.Lock()
	current = "2.2.2.2"
	mu.Unlock()
	time.Sleep(40 * time.Millisecond)

	// Stale read returns the OLD IP immediately (no blocking) and kicks off an
	// asynchronous refresh.
	require.Equal(t, "1.1.1.1", r.resolveOne(context.Background(), "peer"))

	// The background refresh should run and update the cache to the new IP.
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("background refresh did not run")
	}
	require.Eventually(t, func() bool {
		r.mu.Lock()
		defer r.mu.Unlock()
		return r.cache["peer"].ip == "2.2.2.2"
	}, 2*time.Second, 5*time.Millisecond, "background refresh did not update cache to new IP")
}
