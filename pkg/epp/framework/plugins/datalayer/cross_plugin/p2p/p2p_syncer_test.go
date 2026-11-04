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

package p2p

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"runtime"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	attrconcurrency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/concurrency"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
)

const (
	testEndpointID = "vortex/backend-rank-0"
	testNamespace  = "router"
	eventually     = 3 * time.Second
	tick           = 5 * time.Millisecond
)

var testStateKey = fwkdl.StateKey("inflight")

// loads is one replica's live in-flight state, read through a spec like
// inflight-load-producer's.
type loads struct {
	mu     sync.Mutex
	values map[string]int64
}

func newLoads() *loads {
	return &loads{values: map[string]int64{}}
}

func (l *loads) set(endpointID string, requests int64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.values[endpointID] = requests
}

func (l *loads) spec() fwkdl.CrossReplicaSpec {
	return fwkdl.CrossReplicaSpec{
		StateKey: testStateKey,
		Read: func(endpointID string) fwkdl.Cloneable {
			l.mu.Lock()
			defer l.mu.Unlock()
			return &attrconcurrency.InFlightLoad{Requests: l.values[endpointID]}
		},
		Aggregate: func(values []any) any {
			total := &attrconcurrency.InFlightLoad{}
			for _, v := range values {
				if ifl, ok := v.(*attrconcurrency.InFlightLoad); ok {
					total.Requests += ifl.Requests
					total.Tokens += ifl.Tokens
				}
			}
			return total
		},
	}
}

// staticPeers is a PeerSource whose set the test changes.
type staticPeers struct {
	mu    sync.Mutex
	peers []fwkdl.PeerMetadata
}

func (p *staticPeers) Peers() []fwkdl.PeerMetadata {
	p.mu.Lock()
	defer p.mu.Unlock()
	return append([]fwkdl.PeerMetadata(nil), p.peers...)
}

func (p *staticPeers) set(peers ...fwkdl.PeerMetadata) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.peers = peers
}

type replica struct {
	name   string
	syncer *P2PSyncer
	loads  *loads
	peers  *staticPeers
	cancel context.CancelFunc
}

func (r *replica) address() string {
	return r.syncer.listener.Addr().String()
}

func (r *replica) publish(t testing.TB) {
	t.Helper()
	require.Empty(t, r.syncer.SetBatch(context.Background(), r.loads.spec(), []string{testEndpointID}, time.Second))
}

func (r *replica) requests(endpointID string) int64 {
	value, ok, err := r.syncer.Get(context.Background(), r.loads.spec(), endpointID)
	if err != nil || !ok {
		return -1
	}
	return value.(*attrconcurrency.InFlightLoad).Requests
}

func (r *replica) link(peer string) *peerLink {
	r.syncer.linksMu.Lock()
	defer r.syncer.linksMu.Unlock()
	return r.syncer.links[types.NamespacedName{Namespace: testNamespace, Name: peer}]
}

func peerOf(name, address string) fwkdl.PeerMetadata {
	host, port, _ := net.SplitHostPort(address)
	return fwkdl.PeerMetadata{ID: types.NamespacedName{Namespace: testNamespace, Name: name}, Address: host, Port: port}
}

func newReplica(t testing.TB, name string, stateTTL, refreshInterval time.Duration) *replica {
	t.Helper()
	return newReplicaAt(t, name, "127.0.0.1:0", stateTTL, refreshInterval)
}

func newReplicaAt(t testing.TB, name, listenAddress string, stateTTL, refreshInterval time.Duration) *replica {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	syncer, err := newP2PSyncer(ctx, "p2p", name, listenAddress, "", stateTTL, refreshInterval)
	require.NoError(t, err)
	r := &replica{name: name, syncer: syncer, loads: newLoads(), peers: &staticPeers{}, cancel: cancel}
	syncer.SetPeerSource(r.peers)
	t.Cleanup(cancel)
	return r
}

// mesh points every replica at every other one.
func mesh(replicas ...*replica) {
	for _, r := range replicas {
		var peers []fwkdl.PeerMetadata
		for _, other := range replicas {
			if other != r {
				peers = append(peers, peerOf(other.name, other.address()))
			}
		}
		r.peers.set(peers...)
	}
}

// publishUntil publishes from every replica until cond holds, as the
// publisher's tick does.
func publishUntil(t *testing.T, cond func() bool, replicas ...*replica) {
	t.Helper()
	require.Eventually(t, func() bool {
		for _, r := range replicas {
			r.publish(t)
		}
		return cond()
	}, eventually, tick)
}

func TestFactoryConfiguration(t *testing.T) {
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	_, freePort, _ := net.SplitHostPort(listener.Addr().String())
	require.NoError(t, listener.Close())

	tests := []struct {
		name    string
		params  string
		wantErr string
	}{
		{name: "port required", params: `{}`, wantErr: "'port' parameter is required"},
		{name: "port out of range", params: `{"port": "70000"}`, wantErr: "port must be in the range"},
		{name: "unknown field", params: `{"port": "` + freePort + `", "address": "x"}`, wantErr: "invalid parameters"},
		{name: "invalid stateTTL", params: `{"port": "` + freePort + `", "stateTTL": "soon"}`, wantErr: "invalid stateTTL"},
		{name: "refresh not shorter than stateTTL", params: `{"port": "` + freePort + `", "stateTTL": "1s", "refreshInterval": "1s"}`, wantErr: "must be shorter than stateTTL"},
		{name: "valid", params: `{"port": "` + freePort + `", "stateTTL": "3s", "refreshInterval": "50ms"}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plugin, err := P2PSyncerFactory("p2p", json.NewDecoder(strings.NewReader(tt.params)), nil)
			if tt.wantErr != "" {
				require.ErrorContains(t, err, tt.wantErr)
				return
			}
			require.NoError(t, err)
			syncer := plugin.(*P2PSyncer)
			t.Cleanup(syncer.close)
			assert.Equal(t, 3*time.Second, syncer.stateTTL)
			assert.Equal(t, 50*time.Millisecond, syncer.refreshInterval)
			assert.Equal(t, P2PSyncerType, syncer.TypedName().Type)
		})
	}
}

func TestGetReturnsLiveLocalWithoutPeers(t *testing.T) {
	a := newReplica(t, "a", time.Second, 100*time.Millisecond)
	a.loads.set(testEndpointID, 3)
	a.publish(t)
	assert.Equal(t, int64(3), a.requests(testEndpointID))
	a.loads.set(testEndpointID, 4)
	assert.Equal(t, int64(4), a.requests(testEndpointID), "Get reads the live local value")
}

func TestGetCombinesLiveLocalWithPeerValue(t *testing.T) {
	a := newReplica(t, "a", time.Second, 100*time.Millisecond)
	b := newReplica(t, "b", time.Second, 100*time.Millisecond)
	mesh(a, b)
	a.loads.set(testEndpointID, 5)
	b.loads.set(testEndpointID, 2)

	publishUntil(t, func() bool { return b.requests(testEndpointID) == 7 && a.requests(testEndpointID) == 7 }, a, b)
	b.loads.set(testEndpointID, 1)
	assert.Equal(t, int64(6), b.requests(testEndpointID), "the local part of Get is live")
}

func TestValuesReceivedBeforeFirstPublishCountAfterIt(t *testing.T) {
	a := newReplica(t, "a", 5*time.Second, time.Second)
	b := newReplica(t, "b", 5*time.Second, time.Second)
	mesh(a, b)
	a.loads.set(testEndpointID, 5)

	require.Eventually(t, func() bool {
		a.publish(t)
		b.syncer.recvMu.Lock()
		defer b.syncer.recvMu.Unlock()
		return len(b.syncer.received[testStateKey][testEndpointID]) == 1
	}, eventually, tick)
	assert.Equal(t, int64(0), b.requests(testEndpointID), "values of an unregistered key are not decoded")

	b.publish(t)
	assert.Equal(t, int64(5), b.requests(testEndpointID))
}

func TestChangedValueReachesPeerBeforeRefresh(t *testing.T) {
	a := newReplica(t, "a", 30*time.Second, 10*time.Second)
	b := newReplica(t, "b", 30*time.Second, 10*time.Second)
	mesh(a, b)
	b.publish(t)
	a.loads.set(testEndpointID, 1)
	publishUntil(t, func() bool { return b.requests(testEndpointID) == 1 }, a)

	a.loads.set(testEndpointID, 2)
	a.publish(t)
	require.Eventually(t, func() bool { return b.requests(testEndpointID) == 2 }, time.Second, time.Millisecond)
}

func TestUnchangedValueIsResentOnlyAfterRefreshInterval(t *testing.T) {
	a := newReplica(t, "a", 30*time.Second, 10*time.Second)
	b := newReplica(t, "b", 30*time.Second, 10*time.Second)
	mesh(a, b)
	b.publish(t)
	a.loads.set(testEndpointID, 1)
	publishUntil(t, func() bool { return b.requests(testEndpointID) == 1 }, a)

	receivedAt := func() time.Time {
		b.syncer.recvMu.Lock()
		defer b.syncer.recvMu.Unlock()
		return b.syncer.received[testStateKey][testEndpointID][a.syncer.replicaID].receivedAt
	}
	first := receivedAt()
	for range 5 {
		a.publish(t)
	}
	time.Sleep(100 * time.Millisecond)
	assert.Equal(t, first, receivedAt(), "an unchanged value is not resent within refreshInterval")

	a.syncer.sendMu.Lock()
	a.syncer.sent[testStateKey][testEndpointID].sentAt = time.Now().Add(-a.syncer.refreshInterval)
	a.syncer.sendMu.Unlock()
	a.publish(t)
	require.Eventually(t, func() bool { return receivedAt().After(first) }, time.Second, time.Millisecond)
}

func TestPeerValueExpiresAfterStateTTLWhenPeerStops(t *testing.T) {
	a := newReplica(t, "a", 300*time.Millisecond, 50*time.Millisecond)
	b := newReplica(t, "b", 300*time.Millisecond, 50*time.Millisecond)
	mesh(a, b)
	a.loads.set(testEndpointID, 4)
	publishUntil(t, func() bool { return b.requests(testEndpointID) == 4 }, a, b)

	a.cancel()
	stopped := time.Now()
	require.Eventually(t, func() bool { return b.requests(testEndpointID) == 0 }, eventually, tick)
	assert.GreaterOrEqual(t, time.Since(stopped), 250*time.Millisecond, "the value counts until stateTTL after the last message")
}

func TestDeleteRemovesValueFromPeers(t *testing.T) {
	a := newReplica(t, "a", 30*time.Second, 10*time.Second)
	b := newReplica(t, "b", 30*time.Second, 10*time.Second)
	mesh(a, b)
	b.publish(t)
	a.loads.set(testEndpointID, 4)
	publishUntil(t, func() bool { return b.requests(testEndpointID) == 4 }, a)

	require.NoError(t, a.syncer.Delete(context.Background(), testStateKey, testEndpointID))
	require.Eventually(t, func() bool { return b.requests(testEndpointID) == 0 }, time.Second, time.Millisecond)
}

func TestPeerRemovedFromDiscoveryCountsWhileItSends(t *testing.T) {
	a := newReplica(t, "a", 300*time.Millisecond, 50*time.Millisecond)
	b := newReplica(t, "b", 300*time.Millisecond, 50*time.Millisecond)
	mesh(a, b)
	a.loads.set(testEndpointID, 4)
	publishUntil(t, func() bool { return b.requests(testEndpointID) == 4 }, a, b)

	// a drains: b's discovery drops it, a keeps publishing to b.
	b.peers.set()
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		a.publish(t)
		b.publish(t)
		require.Equal(t, int64(4), b.requests(testEndpointID))
		time.Sleep(10 * time.Millisecond)
	}

	a.cancel()
	require.Eventually(t, func() bool { return b.requests(testEndpointID) == 0 }, eventually, tick)
}

func TestJoiningPeerReceivesStateBeforeRefresh(t *testing.T) {
	a := newReplica(t, "a", 30*time.Second, 10*time.Second)
	b := newReplica(t, "b", 30*time.Second, 10*time.Second)
	mesh(a, b)
	a.loads.set(testEndpointID, 1)
	b.loads.set(testEndpointID, 2)
	publishUntil(t, func() bool { return a.requests(testEndpointID) == 3 && b.requests(testEndpointID) == 3 }, a, b)

	c := newReplica(t, "c", 30*time.Second, 10*time.Second)
	mesh(a, b, c)
	publishUntil(t, func() bool { return c.requests(testEndpointID) == 3 }, a, b, c)
}

func TestGetOrSetReturnsCoordinationUnsupported(t *testing.T) {
	a := newReplica(t, "a", time.Second, 100*time.Millisecond)
	_, _, err := a.syncer.GetOrSet(context.Background(), testStateKey, "request-1", "candidate")
	require.ErrorIs(t, err, ErrCoordinationUnsupported)
}

func TestSetBatchReportsCancelledEndpoints(t *testing.T) {
	a := newReplica(t, "a", time.Second, 100*time.Millisecond)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	failures := a.syncer.SetBatch(ctx, a.loads.spec(), []string{"x/a", "x/b"}, time.Second)
	assert.Len(t, failures, 2)
	for _, err := range failures {
		require.ErrorIs(t, err, context.Canceled)
	}
}

func TestSetBatchGetAndReceiveRunConcurrently(t *testing.T) {
	a := newReplica(t, "a", time.Second, 20*time.Millisecond)
	b := newReplica(t, "b", time.Second, 20*time.Millisecond)
	mesh(a, b)
	endpoints := []string{"x/a", "x/b", "x/c", "x/d"}

	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	var wg sync.WaitGroup
	for _, r := range []*replica{a, b} {
		wg.Go(func() {
			for i := int64(0); ctx.Err() == nil; i++ {
				for _, endpointID := range endpoints {
					r.loads.set(endpointID, i%7)
				}
				r.syncer.SetBatch(ctx, r.loads.spec(), endpoints, time.Second)
			}
		})
		wg.Go(func() {
			for ctx.Err() == nil {
				for _, endpointID := range endpoints {
					_, _, _ = r.syncer.Get(ctx, r.loads.spec(), endpointID)
				}
			}
		})
		wg.Go(func() {
			for ctx.Err() == nil {
				_ = r.syncer.Delete(ctx, testStateKey, endpoints[0])
				time.Sleep(time.Millisecond)
			}
		})
	}
	wg.Wait()
}

// TestReplicasConvergeOnTheSum runs several replicas in a full mesh, each with
// its own load on every endpoint, and checks that every replica's Get returns
// the sum across replicas after load changes.
func TestReplicasConvergeOnTheSum(t *testing.T) {
	const replicaCount = 4
	endpoints := []string{"x/a", "x/b", "x/c", "x/d", "x/e", "x/f"}
	replicas := make([]*replica, replicaCount)
	for i := range replicas {
		replicas[i] = newReplica(t, fmt.Sprintf("r%d", i), 2*time.Second, 100*time.Millisecond)
	}
	mesh(replicas...)

	for round := int64(1); round <= 3; round++ {
		want := map[string]int64{}
		for i, r := range replicas {
			for j, endpointID := range endpoints {
				load := round*int64(i+1) + int64(j)
				r.loads.set(endpointID, load)
				want[endpointID] += load
			}
		}
		require.Eventually(t, func() bool {
			for _, r := range replicas {
				require.Empty(t, r.syncer.SetBatch(context.Background(), r.loads.spec(), endpoints, time.Second))
			}
			for _, r := range replicas {
				for _, endpointID := range endpoints {
					if r.requests(endpointID) != want[endpointID] {
						return false
					}
				}
			}
			return true
		}, eventually, tick, "round %d", round)
	}
}

// proxy forwards TCP between a sender and one receiver and can stop
// forwarding to model a link that delivers nothing, as in a network
// partition: bytes stay unacknowledged at the application and the sender's
// buffers fill.
type proxy struct {
	listener net.Listener
	target   string

	mu     sync.Mutex
	cond   *sync.Cond
	paused bool
	conns  []*proxyConn
}

// proxyConn is one proxied connection. A stranded connection stays paused
// after the proxy resumes, as a TCP connection whose retransmissions back off
// past the end of a partition.
type proxyConn struct {
	in, out  net.Conn
	stranded bool
}

func newProxy(t *testing.T, target string) *proxy {
	t.Helper()
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	p := &proxy{listener: listener, target: target}
	p.cond = sync.NewCond(&p.mu)
	go p.accept()
	t.Cleanup(p.close)
	return p
}

func (p *proxy) address() string {
	return p.listener.Addr().String()
}

func (p *proxy) accept() {
	for {
		in, err := p.listener.Accept()
		if err != nil {
			return
		}
		out, err := net.Dial("tcp", p.target)
		if err != nil {
			_ = in.Close()
			continue
		}
		c := &proxyConn{in: in, out: out}
		p.mu.Lock()
		p.conns = append(p.conns, c)
		p.mu.Unlock()
		go p.copy(out, in, c)
		go p.copy(in, out, c)
	}
}

func (p *proxy) copy(dst, src net.Conn, c *proxyConn) {
	buf := make([]byte, 4096)
	for {
		p.mu.Lock()
		for p.paused || c.stranded {
			p.cond.Wait()
		}
		p.mu.Unlock()
		n, err := src.Read(buf)
		if n > 0 {
			p.mu.Lock()
			for p.paused || c.stranded {
				p.cond.Wait()
			}
			p.mu.Unlock()
			if _, werr := dst.Write(buf[:n]); werr != nil {
				return
			}
		}
		if err != nil {
			_ = dst.Close()
			return
		}
	}
}

func (p *proxy) setPaused(paused bool) {
	p.mu.Lock()
	p.paused = paused
	p.cond.Broadcast()
	p.mu.Unlock()
}

// strand pauses the proxy and leaves its current connections paused after it
// resumes; only connections opened later carry data again.
func (p *proxy) strand() {
	p.mu.Lock()
	p.paused = true
	for _, c := range p.conns {
		c.stranded = true
	}
	p.cond.Broadcast()
	p.mu.Unlock()
}

func (p *proxy) close() {
	_ = p.listener.Close()
	p.mu.Lock()
	defer p.mu.Unlock()
	p.paused = false
	for _, c := range p.conns {
		c.stranded = false
		_ = c.in.Close()
		_ = c.out.Close()
	}
	p.cond.Broadcast()
}

func TestPeerLeavingDoesNotPauseOtherPeers(t *testing.T) {
	a := newReplica(t, "a", 30*time.Second, 10*time.Second)
	b := newReplica(t, "b", 30*time.Second, 10*time.Second)
	c := newReplica(t, "c", 30*time.Second, 10*time.Second)
	mesh(a, b, c)
	a.loads.set(testEndpointID, 0)
	b.loads.set(testEndpointID, 1)
	c.loads.set(testEndpointID, 10)
	publishUntil(t, func() bool { return a.requests(testEndpointID) == 11 }, a, b, c)

	c.cancel()
	time.Sleep(20 * time.Millisecond)
	b.loads.set(testEndpointID, 2)
	start := time.Now()
	b.publish(t)
	require.Eventually(t, func() bool { return a.requests(testEndpointID) == 12 }, eventually, time.Millisecond)
	assert.Less(t, time.Since(start), 150*time.Millisecond, "a peer closing its link paused receiving from the others")
}

func TestStalledPeerDoesNotDelayOtherPeers(t *testing.T) {
	a := newReplica(t, "a", 30*time.Second, 10*time.Second)
	b := newReplica(t, "b", 30*time.Second, 10*time.Second)
	stalled := newReplica(t, "stalled", 30*time.Second, 10*time.Second)
	toStalled := newProxy(t, stalled.address())
	a.peers.set(peerOf("b", b.address()), peerOf("stalled", toStalled.address()))
	b.publish(t)
	stalled.publish(t)

	// Large values fill the stalled link's socket buffers quickly.
	big := make([]string, 0, 2001)
	for i := range 2000 {
		big = append(big, fmt.Sprintf("x/endpoint-%04d", i))
	}
	publishUntil(t, func() bool { return b.requests(testEndpointID) == 0 && stalled.requests(testEndpointID) == 0 }, a)
	a.loads.set(testEndpointID, 1)
	publishUntil(t, func() bool { return b.requests(testEndpointID) == 1 && stalled.requests(testEndpointID) == 1 }, a)

	toStalled.setPaused(true)
	for round := int64(2); round < 40; round++ {
		for _, endpointID := range big {
			a.loads.set(endpointID, round)
		}
		a.loads.set(testEndpointID, round)
		start := time.Now()
		require.Empty(t, a.syncer.SetBatch(context.Background(), a.loads.spec(), append(big[:len(big):len(big)], testEndpointID), time.Second))
		require.Less(t, time.Since(start), 500*time.Millisecond, "SetBatch does not wait on a stalled peer")
		require.Eventually(t, func() bool { return b.requests(testEndpointID) == round }, time.Second, time.Millisecond,
			"round %d reaches the healthy peer", round)
	}
	assert.Less(t, stalled.requests(testEndpointID), int64(39), "the stalled peer received nothing new")
}

// TestPartitionedReplicaExpiresAndRecovers runs three replicas with a proxy
// on every directed link, cuts one replica off from the others, and checks
// that the others drop its values after stateTTL while they keep syncing with
// each other, and that all values return after the partition heals.
func TestPartitionedReplicaExpiresAndRecovers(t *testing.T) {
	t.Parallel()
	const stateTTL = 400 * time.Millisecond
	replicas := map[string]*replica{}
	for _, name := range []string{"a", "b", "c"} {
		replicas[name] = newReplica(t, name, stateTTL, 50*time.Millisecond)
	}
	links := map[[2]string]*proxy{}
	for from, sender := range replicas {
		var peers []fwkdl.PeerMetadata
		for to, receiver := range replicas {
			if from == to {
				continue
			}
			link := newProxy(t, receiver.address())
			links[[2]string{from, to}] = link
			peers = append(peers, peerOf(to, link.address()))
		}
		sender.peers.set(peers...)
	}
	a, b, c := replicas["a"], replicas["b"], replicas["c"]
	a.loads.set(testEndpointID, 1)
	b.loads.set(testEndpointID, 10)
	c.loads.set(testEndpointID, 100)
	all := []*replica{a, b, c}
	publishUntil(t, func() bool {
		return a.requests(testEndpointID) == 111 && b.requests(testEndpointID) == 111 && c.requests(testEndpointID) == 111
	}, all...)

	partition := func(cut bool) {
		for key, link := range links {
			if key[0] == "c" || key[1] == "c" {
				if cut {
					link.strand()
				} else {
					link.setPaused(false)
				}
			}
		}
	}
	partition(true)
	cut := time.Now()
	a.loads.set(testEndpointID, 2)
	publishUntil(t, func() bool {
		return a.requests(testEndpointID) == 12 && b.requests(testEndpointID) == 12 && c.requests(testEndpointID) == 100
	}, all...)
	assert.GreaterOrEqual(t, time.Since(cut), stateTTL-50*time.Millisecond, "values count until stateTTL after the last message")

	// The connections open at the cut stay stranded after it heals, so values
	// return only once keepalive has closed them and a new connection is up.
	time.Sleep(time.Until(cut.Add(keepaliveTime + keepaliveTimeout + time.Second)))
	partition(false)
	healed := time.Now()
	publishUntil(t, func() bool {
		return a.requests(testEndpointID) == 112 && b.requests(testEndpointID) == 112 && c.requests(testEndpointID) == 112
	}, all...)
	assert.Less(t, time.Since(healed), 2*redialMaxDelay+dialTimeout, "values return within a redial after the partition heals")
}

func TestUnchangedValueWithDifferentCloneIsNotResent(t *testing.T) {
	a := newReplica(t, "a", 30*time.Second, 10*time.Second)
	// Read returns nil Prompts; Clone of it returns an empty slice.
	spec := fwkdl.CrossReplicaSpec{
		StateKey:  "prefix",
		Read:      func(string) fwkdl.Cloneable { return &attrprefix.PrefixHashSet{} },
		Aggregate: func(values []any) any { return values[0] },
	}
	require.Empty(t, a.syncer.SetBatch(context.Background(), spec, []string{"x/a"}, time.Second))
	sentAt := func() time.Time {
		a.syncer.sendMu.Lock()
		defer a.syncer.sendMu.Unlock()
		return a.syncer.sent[spec.StateKey]["x/a"].sentAt
	}
	first := sentAt()
	time.Sleep(2 * time.Millisecond)
	require.Empty(t, a.syncer.SetBatch(context.Background(), spec, []string{"x/a"}, time.Second))
	assert.Equal(t, first, sentAt(), "an unchanged value is not resent within refreshInterval")
}

func TestOversizedOrMalformedInputDoesNotStopTheReplica(t *testing.T) {
	a := newReplica(t, "a", 30*time.Second, 10*time.Second)
	b := newReplica(t, "b", 30*time.Second, 10*time.Second)

	// A ZMTP greeting and a frame header that claims 2^62 bytes.
	raw, err := net.Dial("tcp", a.address())
	require.NoError(t, err)
	_, _ = raw.Write(append([]byte{0xff, 0, 0, 0, 0, 0, 0, 0, 1, 0x7f, 3, 0, 'N', 'U', 'L', 'L'}, 0x02, 0x40, 0, 0, 0, 0, 0, 0, 0))
	defer func() { _ = raw.Close() }()

	conn, err := grpc.NewClient(a.address(), grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.ForceCodec(frameCodec{})))
	require.NoError(t, err)
	defer func() { _ = conn.Close() }()
	stream, err := conn.NewStream(context.Background(), &publishStream, publishMethod)
	require.NoError(t, err)
	_ = stream.SendMsg(make([]byte, maxMessageBytes+1))
	require.Error(t, stream.RecvMsg(new([]byte)), "the oversized message ends its stream")

	mesh(a, b)
	b.loads.set(testEndpointID, 3)
	publishUntil(t, func() bool { return a.requests(testEndpointID) == 3 }, a, b)
}

func TestIdleConnectionDoesNotBlockPeers(t *testing.T) {
	a := newReplica(t, "a", 30*time.Second, 10*time.Second)
	b := newReplica(t, "b", 30*time.Second, 10*time.Second)
	idle, err := net.Dial("tcp", a.address())
	require.NoError(t, err)
	defer func() { _ = idle.Close() }()

	mesh(a, b)
	b.loads.set(testEndpointID, 7)
	start := time.Now()
	publishUntil(t, func() bool { return a.requests(testEndpointID) == 7 }, a, b)
	assert.Less(t, time.Since(start), time.Second, "a connection that never completes its handshake delayed another peer")
}

func TestRestartedPeerOnSameAddressGetsStateBeforeRefresh(t *testing.T) {
	a := newReplica(t, "a", 30*time.Second, 10*time.Second)
	b := newReplica(t, "b", 30*time.Second, 10*time.Second)
	mesh(a, b)
	a.loads.set(testEndpointID, 5)
	publishUntil(t, func() bool { return b.requests(testEndpointID) == 5 }, a, b)

	address := b.address()
	b.syncer.close()
	restarted := newReplicaAt(t, "b", address, 30*time.Second, 10*time.Second)
	restarted.peers.set(peerOf("a", a.address()))
	start := time.Now()
	publishUntil(t, func() bool { return restarted.requests(testEndpointID) == 5 }, a, restarted)
	assert.Less(t, time.Since(start), 2*redialMaxDelay+dialTimeout, "a's link redials the restarted peer and resends")
}

func TestPeerSetChurnLeaksNoGoroutines(t *testing.T) {
	a := newReplica(t, "a", 30*time.Second, 10*time.Second)
	b := newReplica(t, "b", 30*time.Second, 10*time.Second)
	mesh(a, b)
	b.loads.set(testEndpointID, 1)
	publishUntil(t, func() bool { return a.requests(testEndpointID) == 1 }, a, b)
	baseline := runtime.NumGoroutine()

	for range 20 {
		a.peers.set()
		require.Eventually(t, func() bool { return a.link("b") == nil }, eventually, tick)
		a.peers.set(peerOf("b", b.address()))
		require.Eventually(t, func() bool { l := a.link("b"); return l != nil && l.downSince.Load() == 0 }, eventually, tick)
	}
	require.Eventually(t, func() bool { return runtime.NumGoroutine() <= baseline+2 }, eventually, 10*time.Millisecond,
		"goroutines after 20 link closes: %d, baseline %d", runtime.NumGoroutine(), baseline)
}

// BenchmarkSetToPeerGet measures the time from a changed value's SetBatch on
// one replica to Get returning it on another, over loopback TCP.
func BenchmarkSetToPeerGet(b *testing.B) {
	src := newReplica(b, "a", 30*time.Second, 10*time.Second)
	dst := newReplica(b, "b", 30*time.Second, 10*time.Second)
	mesh(src, dst)
	dst.publish(b)
	src.loads.set(testEndpointID, 1)
	for dst.requests(testEndpointID) != 1 {
		src.publish(b)
		time.Sleep(time.Millisecond)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		want := int64(i + 2)
		src.loads.set(testEndpointID, want)
		for dst.requests(testEndpointID) != want {
			src.publish(b)
		}
	}
}
