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
	"fmt"
	"net"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/go-logr/logr"
	"golang.org/x/sync/singleflight"
)

// envMoRIIODNSResolveTTLSeconds overrides the peer-IP re-resolution TTL (in
// seconds). Operators can widen or tighten the window without a rebuild.
const envMoRIIODNSResolveTTLSeconds = "MORIIO_DNS_RESOLVE_TTL_SECONDS"

// envMoRIIODNSResolveTimeoutSeconds overrides the per-lookup DNS timeout (in
// seconds). Bounds how long a single request-path resolution may block before
// falling back to the last-known-good IP (or the raw spec).
const envMoRIIODNSResolveTimeoutSeconds = "MORIIO_DNS_RESOLVE_TIMEOUT_SECONDS"

// defaultResolveTTL bounds how long a resolved peer IP is cached before the
// next request-time lookup re-resolves it. Kept short so a peer pod that
// restarts with a new IP is picked up within one TTL window, but long enough
// that steady-state request handling does not perform a DNS lookup per call.
const defaultResolveTTL = 30 * time.Second

// defaultResolveTimeout bounds a single DNS lookup on the request path so a
// slow or hanging resolver cannot stall request handling indefinitely. On
// timeout the resolver falls back to the last-known-good IP (or the raw spec).
const defaultResolveTimeout = 5 * time.Second

// resolveTTLFromEnv returns the configured re-resolution TTL, honoring
// MORIIO_DNS_RESOLVE_TTL_SECONDS when set to a positive integer and falling
// back to defaultResolveTTL otherwise.
func resolveTTLFromEnv() time.Duration {
	if v := os.Getenv(envMoRIIODNSResolveTTLSeconds); v != "" {
		if secs, err := strconv.Atoi(v); err == nil && secs > 0 {
			return time.Duration(secs) * time.Second
		}
	}
	return defaultResolveTTL
}

// resolveTimeoutFromEnv returns the configured per-lookup DNS timeout, honoring
// MORIIO_DNS_RESOLVE_TIMEOUT_SECONDS when set to a positive integer and falling
// back to defaultResolveTimeout otherwise.
func resolveTimeoutFromEnv() time.Duration {
	if v := os.Getenv(envMoRIIODNSResolveTimeoutSeconds); v != "" {
		if secs, err := strconv.Atoi(v); err == nil && secs > 0 {
			return time.Duration(secs) * time.Second
		}
	}
	return defaultResolveTimeout
}

// resolvedHost caches the resolution of a single host spec.
type resolvedHost struct {
	ip       string
	resolved time.Time
}

// hostResolver re-resolves DNS host specs to IPs with a short TTL, so a peer
// pod that restarts with a new IP is eventually dialed at its new address
// instead of the router forever emitting the boot-time IP. Raw IP specs pass
// through unchanged and are never looked up. Safe for concurrent use.
//
// This complements the one-shot resolution in Options.Complete(): boot-time
// resolution still validates and seeds the initial IPs, while this resolver
// keeps them fresh on the request path. The sidecar itself does not open the
// RDMA/handshake connection (vLLM does, using the IPs the sidecar injects into
// kv_transfer_params), so a targeted "re-resolve on dial failure" is not
// observable here; a bounded TTL is the low-risk equivalent.
//
// Each request-path lookup is bounded by a context timeout (the smaller of the
// configured timeout and the caller's remaining request deadline), and
// concurrent lookups for the same spec are de-duplicated via a singleflight
// group so a TTL-expiry stampede issues a single DNS query rather than one per
// request. Once a spec has been resolved at least once, subsequent requests are
// served the cached IP immediately (serve-stale-while-revalidate): a past-TTL
// entry is returned without blocking while a single background goroutine
// refreshes it, so only the very first (cold-start) lookup for a spec blocks.
type hostResolver struct {
	ttl     time.Duration
	timeout time.Duration
	logger  logr.Logger

	// lookupIP performs the actual DNS lookup. A field (rather than a direct
	// net.LookupIP call) so tests can inject a deterministic/counting resolver.
	// Defaults to net.DefaultResolver.LookupIPAddr.
	lookupIP func(ctx context.Context, host string) ([]net.IPAddr, error)

	// group de-duplicates concurrent lookups for the same host spec.
	group singleflight.Group

	mu    sync.Mutex
	cache map[string]resolvedHost
	// refreshing tracks specs with a background refresh already in flight, so a
	// staleness burst spawns at most one goroutine per spec.
	refreshing map[string]bool
}

// newHostResolver builds a resolver with the given TTL (falling back to the
// default for non-positive values). The per-lookup timeout is taken from the
// environment (MORIIO_DNS_RESOLVE_TIMEOUT_SECONDS) or the default.
func newHostResolver(logger logr.Logger, ttl time.Duration) *hostResolver {
	if ttl <= 0 {
		ttl = defaultResolveTTL
	}
	registerDNSMetrics()
	return &hostResolver{
		ttl:        ttl,
		timeout:    resolveTimeoutFromEnv(),
		logger:     logger,
		lookupIP:   net.DefaultResolver.LookupIPAddr,
		cache:      map[string]resolvedHost{},
		refreshing: map[string]bool{},
	}
}

// seed pre-populates the cache with a boot-time resolved spec->IP mapping so
// the first request serves the startup IP instead of repeating the lookup, and
// so a request-time DNS outage still yields the last-known-good IP that startup
// resolved (rather than passing the raw hostname through to the MoRI-IO
// handshake). A no-op for empty values, for a spec that is already a literal IP
// (never cached), or when an entry already exists. The seeded entry is marked
// fresh (resolved=now) so it is served without a lookup for one TTL window,
// then refreshed asynchronously via the normal serve-stale path.
func (r *hostResolver) seed(spec, ip string) {
	if spec == "" || ip == "" || net.ParseIP(spec) != nil {
		return
	}
	r.mu.Lock()
	if _, exists := r.cache[spec]; !exists {
		r.cache[spec] = resolvedHost{ip: ip, resolved: time.Now()}
	}
	r.mu.Unlock()
}

// lookup resolves one DNS name to a single IP string using the caller's
// context, preferring IPv4 (falling back to the first returned address). The
// effective deadline is the smaller of the resolver's configured timeout and
// any deadline already on ctx. Callers must handle raw-IP passthrough first.
func (r *hostResolver) lookup(ctx context.Context, host string) (string, error) {
	lookupCtx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	return lookupIPv4Preferred(lookupCtx, r.lookupIP, host)
}

// lookupIPv4Preferred performs a single DNS lookup via lookupIP and returns one
// IP string, preferring IPv4 and falling back to the first returned address.
// The caller supplies the context (and thus the timeout) and must handle raw-IP
// passthrough beforehand. Shared by the boot-time resolveSingleHost and the
// request-path hostResolver.lookup so both apply identical bounding and
// IPv4-preference semantics.
func lookupIPv4Preferred(
	ctx context.Context,
	lookupIP func(ctx context.Context, host string) ([]net.IPAddr, error),
	host string,
) (string, error) {
	addrs, err := lookupIP(ctx, host)
	if err != nil {
		return "", fmt.Errorf("failed to resolve %q: %w", host, err)
	}
	if len(addrs) == 0 {
		return "", fmt.Errorf("no IPs found for %q", host)
	}
	// Prefer IPv4 if available.
	for _, a := range addrs {
		if ipv4 := a.IP.To4(); ipv4 != nil {
			return ipv4.String(), nil
		}
	}
	// Fall back to the first address if no IPv4 found.
	return addrs[0].IP.String(), nil
}

// resolveOne returns the current IP for a single host spec. Raw IPs pass
// through untouched (no lookup, no singleflight, no goroutine). Behavior is
// serve-stale-while-revalidate:
//   - Fresh cache entry (within TTL): returned immediately.
//   - Stale cache entry (past TTL): the stale IP is returned immediately and a
//     single background goroutine re-resolves it (per-spec, bounded by
//     singleflight and the lookup timeout), so no request blocks on DNS after
//     the first successful resolution. A transient failure keeps the
//     last-known-good IP.
//   - No cache entry (cold start): blocks on ctx to perform the first lookup;
//     on hard failure it returns the original spec so the downstream engine
//     surfaces the dial error rather than the router silently dropping the host.
func (r *hostResolver) resolveOne(ctx context.Context, spec string) string {
	// Raw IPs never need resolution.
	if ip := net.ParseIP(spec); ip != nil {
		return spec
	}

	now := time.Now()
	r.mu.Lock()
	entry, ok := r.cache[spec]
	r.mu.Unlock()
	if ok {
		if now.Sub(entry.resolved) < r.ttl {
			return entry.ip
		}
		// Stale: serve the cached IP now, refresh in the background.
		r.triggerRefresh(spec)
		return entry.ip
	}

	// Cold start: block on the caller's context for the first lookup.
	ip, err := r.doLookup(ctx, spec)
	if err != nil {
		r.logger.V(1).Info("MoRI-IO peer DNS resolution failed; passing host through unresolved",
			"host", spec, "error", err.Error())
		return spec
	}
	return ip
}

// doLookup performs a singleflight-coalesced DNS lookup for spec, updates the
// cache, and records metrics/logs. Concurrent callers for the same spec share
// one underlying lookup. Returns the resolved IP or the lookup error.
func (r *hostResolver) doLookup(ctx context.Context, spec string) (string, error) {
	v, err, _ := r.group.Do(spec, func() (interface{}, error) {
		ip, lerr := r.lookup(ctx, spec)
		if lerr != nil {
			dnsLookupFailuresTotal.Inc()
			return "", lerr
		}
		dnsReresolveTotal.Inc()
		r.mu.Lock()
		old, had := r.cache[spec]
		r.cache[spec] = resolvedHost{ip: ip, resolved: time.Now()}
		r.mu.Unlock()
		if had && old.ip != ip {
			dnsIPChangedTotal.Inc()
			r.logger.Info("MoRI-IO peer IP changed on re-resolution",
				"host", spec, "oldIP", old.ip, "newIP", ip)
		}
		return ip, nil
	})
	if err != nil {
		return "", err
	}
	return v.(string), nil
}

// triggerRefresh starts at most one background re-resolution for spec. The
// refresh uses its own (background) context so it is not cancelled when the
// triggering request finishes, but is still bounded by the lookup timeout and
// coalesced by singleflight. On failure the stale cache entry is kept.
func (r *hostResolver) triggerRefresh(spec string) {
	r.mu.Lock()
	if r.refreshing[spec] {
		r.mu.Unlock()
		return
	}
	r.refreshing[spec] = true
	r.mu.Unlock()

	go func() {
		defer func() {
			r.mu.Lock()
			delete(r.refreshing, spec)
			r.mu.Unlock()
		}()
		ctx, cancel := context.WithTimeout(context.Background(), r.timeout)
		defer cancel()
		if _, err := r.doLookup(ctx, spec); err != nil {
			r.logger.V(1).Info("MoRI-IO peer DNS background re-resolution failed; keeping stale IP",
				"host", spec, "error", err.Error())
		}
	}()
}

// resolve returns current IPs for a list of specs, preserving order. An empty
// or nil input is returned unchanged.
func (r *hostResolver) resolve(ctx context.Context, specs []string) []string {
	if len(specs) == 0 {
		return specs
	}
	out := make([]string, len(specs))
	for i, spec := range specs {
		out[i] = r.resolveOne(ctx, spec)
	}
	return out
}
