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
	"errors"
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
	require.Equal(t, testPrefillHostIP1, r.resolveOne(ctx, testPrefillHostIP1))
	require.Equal(t, testLoopbackIPv6, r.resolveOne(ctx, testLoopbackIPv6))
	require.Empty(t, r.cache)

	require.Equal(t, []string{testPrefillHostIP1, testPrefillHostIP2},
		r.resolve(ctx, []string{testPrefillHostIP1, testPrefillHostIP2}))
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
	first := r.resolveOne(ctx, testLocalHostname)
	require.True(t, first == testLoopbackIP || first == testLoopbackIPv6, "unexpected localhost IP %q", first)
	require.Len(t, r.cache, 1)

	entry := r.cache[testLocalHostname]
	// A second call within the TTL must reuse the cached entry (same resolved time).
	second := r.resolveOne(ctx, testLocalHostname)
	require.Equal(t, first, second)
	require.Equal(t, entry.resolved, r.cache[testLocalHostname].resolved)
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

// TestHostResolver_SeedServesStartupIPOnDNSFailure verifies that a resolver
// seeded with the startup-resolved spec->IP serves that IP on the request path
// even when DNS is unavailable, instead of falling back to the raw hostname
// (which would hang the MoRI-IO handshake). This is the request-time DNS
// failure case raised in review.
func TestHostResolver_SeedServesStartupIPOnDNSFailure(t *testing.T) {
	ctx := context.Background()
	r := newHostResolver(logr.Discard(), 30*time.Millisecond)

	var calls int32
	r.lookupIP = func(_ context.Context, _ string) ([]net.IPAddr, error) {
		atomic.AddInt32(&calls, 1)
		return nil, errors.New("dns unavailable")
	}

	const spec = "decode-0.ns.svc"
	const bootIP = "10.0.0.5"
	r.seed(spec, bootIP)

	// Within TTL: served straight from the seeded entry, no lookup attempted
	// even though DNS is down.
	require.Equal(t, bootIP, r.resolveOne(ctx, spec))
	require.Equal(t, int32(0), atomic.LoadInt32(&calls))

	// Past TTL: serve-stale returns the seeded (last-known-good) IP immediately
	// and triggers a background refresh that fails; the good IP is retained.
	time.Sleep(40 * time.Millisecond)
	require.Equal(t, bootIP, r.resolveOne(ctx, spec))
	require.Eventually(t, func() bool { return atomic.LoadInt32(&calls) >= 1 },
		time.Second, 5*time.Millisecond, "background refresh should have attempted a lookup")
	require.Equal(t, bootIP, r.resolveOne(ctx, spec))
}

// TestHostResolver_SeedNoOps verifies seed ignores empty values and literal-IP
// specs (which are never cached).
func TestHostResolver_SeedNoOps(t *testing.T) {
	r := newHostResolver(logr.Discard(), time.Minute)
	r.seed("", testPrefillHostIP1)
	r.seed("host.ns.svc", "")
	r.seed(testPrefillHostIP1, testPrefillHostIP1) // literal IP spec: never cached
	require.Empty(t, r.cache)
}

// TestServer_ResolverSeededFromStartup verifies the Server seeds its request-
// path resolver with the startup-resolved spec->IP values, so the first request
// serves the boot IP even if DNS has since gone down (rather than the raw spec).
func TestServer_ResolverSeededFromStartup(t *testing.T) {
	s := &Server{
		config: Config{
			MoRIIODecodePodIP:     "10.0.0.9",
			MoRIIODecodePodIPSpec: "localpod.ns.svc",
			MoRIIODecodeHosts:     []string{testDecodeHostIP2, testDecodeHostIP3},
			MoRIIODecodeHostSpecs: []string{"decode-0.ns.svc", "decode-1.ns.svc"},
		},
	}
	// Trigger lazy creation + seeding, then simulate DNS being unavailable.
	r := s.resolver()
	r.lookupIP = func(_ context.Context, _ string) ([]net.IPAddr, error) {
		return nil, errors.New("dns unavailable")
	}
	// The request path must return the startup-resolved IPs, not the raw specs.
	require.Equal(t, "10.0.0.9", s.currentDecodePodIP(context.Background()))
	require.Equal(t, []string{testDecodeHostIP2, testDecodeHostIP3},
		s.currentDecodeHosts(context.Background()))
}
