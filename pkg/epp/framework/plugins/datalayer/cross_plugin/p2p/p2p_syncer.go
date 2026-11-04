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
	"bytes"
	"context"
	"encoding/gob"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"net"
	"os"
	"reflect"
	"slices"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
	"golang.org/x/time/rate"
	"google.golang.org/grpc"
	"google.golang.org/grpc/backoff"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/keepalive"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const (
	P2PSyncerType          = "p2p-syncer"
	defaultStateTTL        = 2 * time.Second
	defaultRefreshInterval = 100 * time.Millisecond
	minimumDuration        = time.Millisecond
	// maintenanceInterval paces peer link reconciliation, heartbeats and
	// expiry of peer values.
	maintenanceInterval = 50 * time.Millisecond
	// heartbeatKey is the state key of the empty message a replica sends when
	// it has sent nothing for refreshInterval, so peers can tell it is
	// reachable.
	heartbeatKey = ""
	// sendQueueLimit bounds the messages queued for one peer. Messages beyond
	// it are dropped; each value is resent within refreshInterval.
	sendQueueLimit = 256
	// maxMessageBytes bounds one message in both directions.
	maxMessageBytes = 16 << 20
	// A connection whose peer acknowledges nothing for keepaliveTime plus
	// keepaliveTimeout is closed and redialed. 10s is the shortest client
	// keepalive time gRPC allows.
	keepaliveTime    = 10 * time.Second
	keepaliveTimeout = 5 * time.Second
	// handshakeTimeout bounds the setup of one incoming connection.
	handshakeTimeout = 5 * time.Second
	dialTimeout      = time.Second
	redialMaxDelay   = time.Second
	errorLogPeriod   = 30 * time.Second
	publishMethod    = "/llmd.epp.p2psyncer.v1.PeerSync/Publish"
)

// publishStream is the one RPC between replicas: a stream of messages from
// the replica that dials to the replica that listens.
var publishStream = grpc.StreamDesc{StreamName: "Publish", ClientStreams: true}

// ErrCoordinationUnsupported is returned by GetOrSet. Peer-to-peer sync has no
// store that every replica writes through, so it cannot provide the
// linearizable request-level coordination GetOrSet requires.
var ErrCoordinationUnsupported = errors.New("p2p-syncer: GetOrSet requires a linearizable store shared by every replica")

var _ fwkdl.CrossReplicaSyncer = (*P2PSyncer)(nil)
var _ fwkdl.CrossReplicaBatchSyncer = (*P2PSyncer)(nil)
var _ fwkdl.PeerConsumer = (*P2PSyncer)(nil)

type p2pConfig struct {
	Port            string `json:"port"`
	StateTTL        string `json:"stateTTL"`
	RefreshInterval string `json:"refreshInterval"`
}

// P2PSyncer is a CrossReplicaSyncer that exchanges endpoint state directly
// between EPP replicas over gRPC. Each replica serves one listener and sends
// to every discovered peer on a stream of its own per peer. Received peer
// values update the peer aggregate used by Get on arrival.
type P2PSyncer struct {
	typedName       fwkplugin.TypedName
	replicaID       string
	podName         string
	port            string
	stateTTL        time.Duration
	refreshInterval time.Duration

	ctx       context.Context
	stop      context.CancelFunc
	closeOnce sync.Once
	listener  net.Listener
	server    *grpc.Server
	source    atomic.Pointer[peerSourceRef]

	linksMu sync.Mutex
	links   map[types.NamespacedName]*peerLink
	// linkList is the current set of links, read without linksMu on publish.
	linkList atomic.Pointer[[]*peerLink]

	sendMu sync.Mutex
	sent   map[fwkdl.StateKey]map[string]*sentValue

	recvMu   sync.Mutex
	specs    map[fwkdl.StateKey]*keySpec
	received map[fwkdl.StateKey]map[string]map[string]*receivedValue
	// heard records when each peer pod last delivered a message.
	heard map[string]time.Time

	cache sync.Map

	// lastSend is the UnixNano time of the last message sent to peers.
	lastSend atomic.Int64

	errorLog rate.Sometimes
}

type peerSourceRef struct {
	fwkdl.PeerSource
}

type keySpec struct {
	prototype any
	aggregate func([]any) any
}

type sentValue struct {
	data   []byte
	sentAt time.Time
}

type receivedValue struct {
	raw        []byte
	value      any
	receivedAt time.Time
}

type aggregateCacheEntry struct {
	value any
}

// envelope is one message from a replica: values it publishes for one state
// key, and endpoints it no longer publishes.
type envelope struct {
	Key     string
	Replica string
	Pod     string
	Values  map[string][]byte
	Deleted []string
}

type peerLink struct {
	id      types.NamespacedName
	address string
	conn    *grpc.ClientConn
	queue   chan []byte
	cancel  context.CancelFunc
	// downSince is when the link last lost its stream, in Unix nanoseconds,
	// or 0 while the stream is open.
	downSince atomic.Int64
	// reported is set once a link down for stateTTL has been logged.
	reported bool
}

// frameCodec carries messages that are already encoded.
type frameCodec struct{}

func (frameCodec) Marshal(v any) ([]byte, error) {
	data, ok := v.([]byte)
	if !ok {
		return nil, fmt.Errorf("p2p-syncer: cannot marshal %T", v)
	}
	return data, nil
}

func (frameCodec) Unmarshal(data []byte, v any) error {
	target, ok := v.(*[]byte)
	if !ok {
		return fmt.Errorf("p2p-syncer: cannot unmarshal into %T", v)
	}
	*target = append([]byte(nil), data...)
	return nil
}

func (frameCodec) Name() string { return P2PSyncerType }

func parseDuration(name, raw string, defaultValue time.Duration) (time.Duration, error) {
	if raw == "" {
		return defaultValue, nil
	}
	d, err := time.ParseDuration(raw)
	if err != nil {
		return 0, fmt.Errorf("p2p-syncer: invalid %s %q: %w", name, raw, err)
	}
	if d < minimumDuration {
		return 0, fmt.Errorf("p2p-syncer: %s must be at least %s, got %s", name, minimumDuration, d)
	}
	return d, nil
}

func P2PSyncerFactory(name string, params *json.Decoder, handle fwkplugin.Handle) (fwkplugin.Plugin, error) {
	var cfg p2pConfig
	if params != nil {
		params.DisallowUnknownFields()
		if err := params.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("p2p-syncer: invalid parameters: %w", err)
		}
	}
	if cfg.Port == "" {
		return nil, errors.New("p2p-syncer: 'port' parameter is required")
	}
	if port, err := strconv.Atoi(cfg.Port); err != nil || port < 1 || port > 65535 {
		return nil, fmt.Errorf("p2p-syncer: port must be in the range 1 through 65535, got %q", cfg.Port)
	}
	stateTTL, err := parseDuration("stateTTL", cfg.StateTTL, defaultStateTTL)
	if err != nil {
		return nil, err
	}
	refreshInterval, err := parseDuration("refreshInterval", cfg.RefreshInterval, defaultRefreshInterval)
	if err != nil {
		return nil, err
	}
	if refreshInterval >= stateTTL {
		return nil, fmt.Errorf("p2p-syncer: refreshInterval %s must be shorter than stateTTL %s", refreshInterval, stateTTL)
	}
	ctx := context.Background()
	if handle != nil && handle.Context() != nil {
		ctx = handle.Context()
	}
	podName, _ := os.Hostname()
	if podName == "" {
		podName = "unknown"
	}
	return newP2PSyncer(ctx, name, podName, ":"+cfg.Port, cfg.Port, stateTTL, refreshInterval)
}

// newP2PSyncer listens on listenAddress and starts the server and the
// maintenance loop, which stop when ctx ends. podName must match the name
// peers discover this replica under. port is the default for peers whose
// metadata carries none.
func newP2PSyncer(ctx context.Context, name, podName, listenAddress, port string, stateTTL, refreshInterval time.Duration) (*P2PSyncer, error) {
	listener, err := net.Listen("tcp", listenAddress)
	if err != nil {
		return nil, fmt.Errorf("p2p-syncer: listen on %s: %w", listenAddress, err)
	}
	ctx, stop := context.WithCancel(ctx)
	s := &P2PSyncer{
		typedName:       fwkplugin.TypedName{Type: P2PSyncerType, Name: name},
		replicaID:       podName + "-" + uuid.NewString(),
		podName:         podName,
		port:            port,
		stateTTL:        stateTTL,
		refreshInterval: refreshInterval,
		ctx:             ctx,
		stop:            stop,
		listener:        listener,
		links:           map[types.NamespacedName]*peerLink{},
		sent:            map[fwkdl.StateKey]map[string]*sentValue{},
		specs:           map[fwkdl.StateKey]*keySpec{},
		received:        map[fwkdl.StateKey]map[string]map[string]*receivedValue{},
		heard:           map[string]time.Time{},
		errorLog:        rate.Sometimes{Interval: errorLogPeriod},
	}
	s.linkList.Store(&[]*peerLink{})
	s.server = grpc.NewServer(
		grpc.ForceServerCodec(frameCodec{}),
		grpc.MaxRecvMsgSize(maxMessageBytes),
		grpc.ConnectionTimeout(handshakeTimeout),
		grpc.KeepaliveParams(keepalive.ServerParameters{Time: keepaliveTime, Timeout: keepaliveTimeout}),
		grpc.KeepaliveEnforcementPolicy(keepalive.EnforcementPolicy{MinTime: keepaliveTime / 2, PermitWithoutStream: true}),
	)
	s.server.RegisterService(&grpc.ServiceDesc{
		ServiceName: "llmd.epp.p2psyncer.v1.PeerSync",
		HandlerType: (*any)(nil),
		Streams: []grpc.StreamDesc{{
			StreamName:    publishStream.StreamName,
			Handler:       func(_ any, stream grpc.ServerStream) error { return s.receive(stream) },
			ClientStreams: true,
		}},
	}, s)
	go func() { _ = s.server.Serve(listener) }()
	go s.maintain()
	context.AfterFunc(ctx, s.close)
	return s, nil
}

func (s *P2PSyncer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

// SetPeerSource supplies the discovered peer set. Peers are read on every
// maintenance tick, so peers that join or leave later are picked up.
func (s *P2PSyncer) SetPeerSource(source fwkdl.PeerSource) {
	s.source.Store(&peerSourceRef{PeerSource: source})
}

func (s *P2PSyncer) cacheKey(key fwkdl.StateKey, endpointID string) string {
	return string(key) + ":" + endpointID
}

func encodeValue(value any) ([]byte, error) {
	if value == nil {
		return nil, errors.New("cannot encode nil value")
	}
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(value); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

func gobDecode(data []byte, prototype any) (any, error) {
	if prototype == nil {
		return nil, errors.New("cannot decode stored value without a type prototype")
	}
	target := reflect.New(reflect.TypeOf(prototype))
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(target.Interface()); err != nil {
		return nil, err
	}
	return target.Elem().Interface(), nil
}

func (s *P2PSyncer) Set(ctx context.Context, spec fwkdl.CrossReplicaSpec, endpointID string) error {
	return s.SetBatch(ctx, spec, []string{endpointID}, 0)[endpointID]
}

// SetBatch sends each endpoint whose value changed since it was last sent, or
// that was last sent refreshInterval or longer ago, in one message per call.
// It never waits on a peer, so timeout is unused.
func (s *P2PSyncer) SetBatch(ctx context.Context, spec fwkdl.CrossReplicaSpec, endpointIDs []string, _ time.Duration) map[string]error {
	failures := make(map[string]error)
	if len(endpointIDs) == 0 {
		return failures
	}
	s.registerSpec(spec, endpointIDs[0])

	now := time.Now()
	values := make(map[string][]byte)
	s.sendMu.Lock()
	sent := s.sent[spec.StateKey]
	if sent == nil {
		sent = make(map[string]*sentValue)
		s.sent[spec.StateKey] = sent
	}
	for _, endpointID := range endpointIDs {
		if err := ctx.Err(); err != nil {
			failures[endpointID] = err
			continue
		}
		data, err := encodeValue(spec.Read(endpointID))
		if err != nil {
			failures[endpointID] = fmt.Errorf("p2p-syncer: encode: %w", err)
			continue
		}
		if prev, ok := sent[endpointID]; ok && now.Sub(prev.sentAt) < s.refreshInterval && bytes.Equal(prev.data, data) {
			continue
		}
		values[endpointID] = data
		sent[endpointID] = &sentValue{data: data, sentAt: now}
	}
	s.sendMu.Unlock()

	if len(values) == 0 {
		return failures
	}
	if err := s.broadcast(spec.StateKey, envelope{Values: values}); err != nil {
		for endpointID := range values {
			failures[endpointID] = err
		}
	}
	if v := ctrl.LoggerFrom(ctx).V(logutil.TRACE); v.Enabled() {
		v.Info("p2p-syncer: SetBatch", "key", string(spec.StateKey), "sent", len(values), "endpoints", len(endpointIDs))
	}
	return failures
}

// broadcast queues one message for every peer link. A peer that does not
// read its stream fills only its own queue.
func (s *P2PSyncer) broadcast(key fwkdl.StateKey, env envelope) error {
	env.Key = string(key)
	env.Replica = s.replicaID
	env.Pod = s.podName
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(env); err != nil {
		return fmt.Errorf("p2p-syncer: encode message: %w", err)
	}
	if buf.Len() > maxMessageBytes {
		return fmt.Errorf("p2p-syncer: message of %d bytes exceeds %d", buf.Len(), maxMessageBytes)
	}
	msg := buf.Bytes()
	s.lastSend.Store(time.Now().UnixNano())
	for _, link := range *s.linkList.Load() {
		select {
		case link.queue <- msg:
		default:
			s.errorLog.Do(func() {
				ctrl.LoggerFrom(s.ctx).Error(errors.New("send queue full"), "p2p-syncer: dropped a message", "peer", link.id.String())
			})
		}
	}
	return nil
}

func (s *P2PSyncer) Get(_ context.Context, spec fwkdl.CrossReplicaSpec, endpointID string) (any, bool, error) {
	local := spec.Read(endpointID)
	cached, ok := s.cache.Load(s.cacheKey(spec.StateKey, endpointID))
	if !ok {
		return local, true, nil
	}
	return spec.Aggregate([]any{local, cached.(*aggregateCacheEntry).value}), true, nil
}

// Delete stops publishing endpointID and tells peers to drop this replica's
// value for it. A peer that misses the message drops the value after stateTTL.
func (s *P2PSyncer) Delete(_ context.Context, key fwkdl.StateKey, endpointID string) error {
	s.sendMu.Lock()
	delete(s.sent[key], endpointID)
	s.sendMu.Unlock()
	return s.broadcast(key, envelope{Deleted: []string{endpointID}})
}

func (s *P2PSyncer) GetOrSet(context.Context, fwkdl.StateKey, string, any) (any, bool, error) {
	return nil, false, ErrCoordinationUnsupported
}

// registerSpec records the value type and aggregate for key, and decodes
// values received before the first local publish.
func (s *P2PSyncer) registerSpec(spec fwkdl.CrossReplicaSpec, endpointID string) {
	s.recvMu.Lock()
	defer s.recvMu.Unlock()
	if _, ok := s.specs[spec.StateKey]; ok {
		return
	}
	prototype := spec.Read(endpointID)
	if prototype == nil {
		return
	}
	ks := &keySpec{prototype: prototype, aggregate: spec.Aggregate}
	s.specs[spec.StateKey] = ks
	for id, byReplica := range s.received[spec.StateKey] {
		for replica, rv := range byReplica {
			if err := s.decodeInto(rv, ks); err != nil {
				delete(byReplica, replica)
			}
		}
		s.refreshAggregateLocked(spec.StateKey, id)
	}
}

func (s *P2PSyncer) decodeInto(rv *receivedValue, ks *keySpec) error {
	value, err := gobDecode(rv.raw, ks.prototype)
	if err != nil {
		if v := ctrl.LoggerFrom(s.ctx).V(logutil.DEBUG); v.Enabled() {
			v.Info("p2p-syncer: decode error", "error", err)
		}
		return err
	}
	rv.value = value
	return nil
}

// receive applies the messages of one peer's stream until the stream ends.
func (s *P2PSyncer) receive(stream grpc.ServerStream) error {
	for {
		var msg []byte
		if err := stream.RecvMsg(&msg); err != nil {
			return nil
		}
		var env envelope
		if err := gob.NewDecoder(bytes.NewReader(msg)).Decode(&env); err != nil {
			s.errorLog.Do(func() { ctrl.LoggerFrom(s.ctx).Error(err, "p2p-syncer: decode message failed") })
			continue
		}
		s.apply(fwkdl.StateKey(env.Key), env, time.Now())
	}
}

// apply stores a peer's values and refreshes the aggregates they feed.
func (s *P2PSyncer) apply(key fwkdl.StateKey, env envelope, now time.Time) {
	if env.Replica == s.replicaID {
		return
	}
	s.recvMu.Lock()
	defer s.recvMu.Unlock()
	if _, ok := s.heard[env.Pod]; !ok {
		ctrl.LoggerFrom(s.ctx).Info("p2p-syncer: receiving from peer", "pod", env.Pod, "replica", env.Replica)
	}
	s.heard[env.Pod] = now
	if len(env.Values) == 0 && len(env.Deleted) == 0 {
		return
	}
	byEndpoint := s.received[key]
	if byEndpoint == nil {
		byEndpoint = make(map[string]map[string]*receivedValue)
		s.received[key] = byEndpoint
	}
	ks := s.specs[key]
	for endpointID, raw := range env.Values {
		rv := &receivedValue{raw: raw, receivedAt: now}
		if ks != nil && s.decodeInto(rv, ks) != nil {
			continue
		}
		byReplica := byEndpoint[endpointID]
		if byReplica == nil {
			byReplica = make(map[string]*receivedValue)
			byEndpoint[endpointID] = byReplica
		}
		byReplica[env.Replica] = rv
		s.refreshAggregateLocked(key, endpointID)
	}
	for _, endpointID := range env.Deleted {
		if byReplica, ok := byEndpoint[endpointID]; ok {
			delete(byReplica, env.Replica)
			if len(byReplica) == 0 {
				delete(byEndpoint, endpointID)
			}
			s.refreshAggregateLocked(key, endpointID)
		}
	}
}

// refreshAggregateLocked recomputes the peer aggregate for one endpoint from
// the decoded peer values, in replica order. recvMu must be held.
func (s *P2PSyncer) refreshAggregateLocked(key fwkdl.StateKey, endpointID string) {
	cacheKey := s.cacheKey(key, endpointID)
	ks := s.specs[key]
	byReplica := s.received[key][endpointID]
	if ks == nil || len(byReplica) == 0 {
		s.cache.Delete(cacheKey)
		return
	}
	values := make([]any, 0, len(byReplica))
	for _, replica := range slices.Sorted(maps.Keys(byReplica)) {
		if v := byReplica[replica].value; v != nil {
			values = append(values, v)
		}
	}
	if len(values) == 0 {
		s.cache.Delete(cacheKey)
		return
	}
	s.cache.Store(cacheKey, &aggregateCacheEntry{value: ks.aggregate(values)})
}

func (s *P2PSyncer) maintain() {
	ticker := time.NewTicker(maintenanceInterval)
	defer ticker.Stop()
	for {
		select {
		case <-s.ctx.Done():
			return
		case <-ticker.C:
			now := time.Now()
			s.expire(now)
			s.reconcileLinks(now)
			if now.Sub(time.Unix(0, s.lastSend.Load())) >= s.refreshInterval {
				_ = s.broadcast(heartbeatKey, envelope{})
			}
		}
	}
}

// expire drops peer values not refreshed within stateTTL. A peer that stops
// sending, whether it exited or is unreachable, stops counting stateTTL after
// its last message.
func (s *P2PSyncer) expire(now time.Time) {
	s.recvMu.Lock()
	defer s.recvMu.Unlock()
	for key, byEndpoint := range s.received {
		for endpointID, byReplica := range byEndpoint {
			expired := false
			for replica, rv := range byReplica {
				if now.Sub(rv.receivedAt) > s.stateTTL {
					delete(byReplica, replica)
					expired = true
				}
			}
			if len(byReplica) == 0 {
				delete(byEndpoint, endpointID)
			}
			if expired {
				s.refreshAggregateLocked(key, endpointID)
			}
		}
	}
	for pod, at := range s.heard {
		if now.Sub(at) > s.stateTTL {
			delete(s.heard, pod)
			ctrl.LoggerFrom(s.ctx).Info("p2p-syncer: no message from peer for stateTTL", "pod", pod, "stateTTL", s.stateTTL)
		}
	}
}

// reconcileLinks opens a link to each discovered peer and closes links to
// peers that left the peer set. Peer values are not dropped with the link: a
// replica that leaves the set while it drains keeps sending, and its values
// expire stateTTL after its last message.
func (s *P2PSyncer) reconcileLinks(now time.Time) {
	ref := s.source.Load()
	if ref == nil {
		return
	}
	desired := make(map[types.NamespacedName]string)
	for _, peer := range ref.Peers() {
		port := peer.Port
		if port == "" {
			port = s.port
		}
		desired[peer.ID] = net.JoinHostPort(peer.Address, port)
	}

	s.linksMu.Lock()
	defer s.linksMu.Unlock()
	changed := false
	for id, link := range s.links {
		if address, ok := desired[id]; ok && address == link.address {
			continue
		}
		link.close()
		delete(s.links, id)
		changed = true
		ctrl.LoggerFrom(s.ctx).Info("p2p-syncer: closed peer link", "peer", id.String(), "address", link.address)
	}
	for id, address := range desired {
		link, ok := s.links[id]
		if !ok {
			var err error
			if link, err = s.dial(id, address, now); err != nil {
				s.errorLog.Do(func() { ctrl.LoggerFrom(s.ctx).Error(err, "p2p-syncer: open peer link failed", "peer", id.String()) })
				continue
			}
			s.links[id] = link
			changed = true
			ctrl.LoggerFrom(s.ctx).Info("p2p-syncer: opened peer link", "peer", id.String(), "address", address)
		}
		down := link.downSince.Load()
		if down == 0 {
			link.reported = false
		} else if !link.reported && now.Sub(time.Unix(0, down)) > s.stateTTL {
			link.reported = true
			ctrl.LoggerFrom(s.ctx).Info("p2p-syncer: peer link not connected", "peer", id.String(), "address", address,
				"since", time.Unix(0, down), "state", link.conn.GetState().String())
		}
	}
	if changed {
		list := slices.Collect(maps.Values(s.links))
		s.linkList.Store(&list)
	}
}

// dial opens a client connection to one peer and starts the link's sender.
// gRPC redials a lost connection by itself, and keepalive closes one that no
// longer delivers, as after a network partition.
func (s *P2PSyncer) dial(id types.NamespacedName, address string, now time.Time) (*peerLink, error) {
	conn, err := grpc.NewClient(address,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithDefaultCallOptions(grpc.ForceCodec(frameCodec{}), grpc.MaxCallSendMsgSize(maxMessageBytes)),
		grpc.WithKeepaliveParams(keepalive.ClientParameters{Time: keepaliveTime, Timeout: keepaliveTimeout, PermitWithoutStream: true}),
		grpc.WithConnectParams(grpc.ConnectParams{
			Backoff:           backoff.Config{BaseDelay: 50 * time.Millisecond, Multiplier: 1.6, Jitter: 0.2, MaxDelay: redialMaxDelay},
			MinConnectTimeout: dialTimeout,
		}),
	)
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithCancel(s.ctx)
	link := &peerLink{id: id, address: address, conn: conn, queue: make(chan []byte, sendQueueLimit), cancel: cancel}
	link.downSince.Store(now.UnixNano())
	go s.send(ctx, link)
	return link, nil
}

// send keeps one stream open to the link's peer and writes the link's queue
// to it. Every value is resent on the next publish after the stream opens.
func (s *P2PSyncer) send(ctx context.Context, link *peerLink) {
	for ctx.Err() == nil {
		if err := s.stream(ctx, link); err != nil && ctx.Err() == nil {
			ctrl.LoggerFrom(s.ctx).Info("p2p-syncer: peer link lost", "peer", link.id.String(), "error", err.Error())
			select {
			case <-ctx.Done():
			case <-time.After(redialMaxDelay):
			}
		}
	}
}

// stream opens one stream, waiting until the connection is ready, and writes
// to it until it fails. The peer never replies, so RecvMsg returns only when
// the stream ends, which ends a stream that has nothing to send.
func (s *P2PSyncer) stream(ctx context.Context, link *peerLink) error {
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	stream, err := link.conn.NewStream(ctx, &publishStream, publishMethod, grpc.WaitForReady(true))
	if err != nil {
		return err
	}
	ended := make(chan error, 1)
	go func() { ended <- stream.RecvMsg(new([]byte)) }()
	link.downSince.Store(0)
	s.resendAll()
	ctrl.LoggerFrom(s.ctx).Info("p2p-syncer: peer link connected", "peer", link.id.String())
	defer link.downSince.Store(time.Now().UnixNano())
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case err := <-ended:
			return fmt.Errorf("stream ended: %w", err)
		case msg := <-link.queue:
			if err := stream.SendMsg(msg); err != nil {
				return err
			}
		}
	}
}

func (s *P2PSyncer) resendAll() {
	s.sendMu.Lock()
	defer s.sendMu.Unlock()
	for _, sent := range s.sent {
		for _, sv := range sent {
			sv.sentAt = time.Time{}
		}
	}
}

func (l *peerLink) close() {
	l.cancel()
	_ = l.conn.Close()
}

// close stops the syncer. It runs when the context passed to newP2PSyncer
// ends.
func (s *P2PSyncer) close() {
	s.closeOnce.Do(func() {
		s.stop()
		s.server.Stop()
		s.linksMu.Lock()
		defer s.linksMu.Unlock()
		for id, link := range s.links {
			link.close()
			delete(s.links, id)
		}
		s.linkList.Store(&[]*peerLink{})
	})
}
