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

package preciseprefixcache

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strconv"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
)

// readServerInfo queries an engine rank's GetServerInfo server and returns
// that rank's global DP rank and whether it publishes KV events. A source
// entry belongs to the rank only when its dp rank matches the responding
// server's own, so a server reporting another rank's sources yields
// publishes=false. An error means the server could not be asked at all.
func readServerInfo(ctx context.Context, client apiv1.SglangServiceClient) (dpRank uint32, publishes bool, err error) {
	resp, err := client.GetServerInfo(ctx, &apiv1.GetServerInfoRequest{})
	if err != nil {
		return 0, false, fmt.Errorf("get server info: %w", err)
	}
	di := resp.GetDeploymentInfo()
	if di == nil || di.DpRank == nil {
		return 0, false, errors.New("server info without deployment dp rank")
	}
	for _, src := range di.GetKvEventSources() {
		if src.GetDpRank() == di.GetDpRank() {
			return di.GetDpRank(), true, nil
		}
	}
	return di.GetDpRank(), false, nil
}

// rankQueryTimeout bounds one GetServerInfo query. Extract runs on endpoint
// lifecycle events, so an unresponsive engine delays discovery by at most
// this long per endpoint.
const rankQueryTimeout = 2 * time.Second

// queryRankServer asks the endpoint's rank GetServerInfo server (listening at
// GrpcBasePort + RankIndex) for its global DP rank and whether it publishes
// KV events. The connection is per call: endpoint events are rare compared
// to requests.
func (p *Producer) queryRankServer(ctx context.Context, meta *fwkdl.EndpointMetadata) (uint32, bool, error) {
	addr := net.JoinHostPort(meta.Address,
		strconv.Itoa(p.kvEventsConfig.PodDiscoveryConfig.GrpcBasePort+meta.GetRankIndex()))
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		return 0, false, fmt.Errorf("dial %s: %w", addr, err)
	}
	defer func() { _ = conn.Close() }()
	queryCtx, cancel := context.WithTimeout(ctx, rankQueryTimeout)
	defer cancel()
	return readServerInfo(queryCtx, apiv1.NewSglangServiceClient(conn))
}

func (p *Producer) ensureGRPCSubscriber(ctx context.Context, meta *fwkdl.EndpointMetadata) error {
	endpointKey := meta.ID.String()

	lookupCtx, cancel := context.WithCancel(p.subscriberCtx)
	lookup := &rankLookup{
		ctx:      lookupCtx,
		cancel:   cancel,
		metadata: meta.Clone(),
	}

	p.rankMu.Lock()
	p.cancelRankLookupLocked(endpointKey)
	if p.rankLookups == nil {
		p.rankLookups = make(map[string]*rankLookup)
	}
	p.rankLookups[endpointKey] = lookup
	p.rankMu.Unlock()

	dpRank, publishes, err := p.queryRankServer(ctx, lookup.metadata)
	if err != nil {
		p.rankMu.Lock()
		shouldRetry := lookup.ctx.Err() == nil &&
			p.rankLookups[endpointKey] == lookup
		p.rankMu.Unlock()

		if shouldRetry {
			log.FromContext(ctx).WithName(p.typedName.String()).Error(
				err, "GetServerInfo failed, retrying", "endpoint", endpointKey)
			go p.retryGRPCSubscriber(lookup)
		}
		return nil
	}

	return p.finishGRPCSubscriber(lookup, dpRank, publishes)
}

// cancelRankLookupLocked requires rankMu.
func (p *Producer) cancelRankLookupLocked(endpointKey string) {
	if lookup := p.rankLookups[endpointKey]; lookup != nil {
		delete(p.rankLookups, endpointKey)
		lookup.cancel()
	}
}

func (p *Producer) retryGRPCSubscriber(lookup *rankLookup) {
	ticker := time.NewTicker(time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-lookup.ctx.Done():
			return
		case <-ticker.C:
		}

		dpRank, publishes, err := p.queryRankServer(lookup.ctx, lookup.metadata)
		if err != nil {
			continue
		}
		if err := p.finishGRPCSubscriber(lookup, dpRank, publishes); err != nil {
			log.FromContext(lookup.ctx).WithName(p.typedName.String()).Error(
				err, "Failed to ensure KV-events subscriber", "endpoint", lookup.metadata.ID.String())
		}
		return
	}
}

func (p *Producer) finishGRPCSubscriber(
	lookup *rankLookup, dpRank uint32, publishes bool,
) error {
	p.rankMu.Lock()
	defer p.rankMu.Unlock()

	meta := lookup.metadata
	endpointKey := meta.ID.String()
	if p.rankLookups[endpointKey] != lookup || lookup.ctx.Err() != nil {
		return nil
	}
	delete(p.rankLookups, endpointKey)
	defer lookup.cancel()

	logger := log.FromContext(lookup.ctx).WithName(p.typedName.String())
	if !publishes {
		logger.V(logging.DEBUG).Info("Rank publishes no KV events, skipping subscriber",
			"endpoint", endpointKey, "dpRank", dpRank)
		p.subscribersManager.RemoveSubscriber(lookup.ctx, endpointKey)
		return nil
	}

	_, replayEndpoint, snapshotEndpoint := p.subscriberEndpoints(meta)
	zmqEndpoint := "tcp://" + net.JoinHostPort(meta.Address,
		strconv.Itoa(p.kvEventsConfig.PodDiscoveryConfig.SocketPort+int(dpRank)))
	sourceEndpoint := fmt.Sprintf("%s:%s", meta.Address, meta.Port)

	if err := p.subscribersManager.EnsureSubscriber(p.subscriberCtx, endpointKey,
		sourceEndpoint, zmqEndpoint, replayEndpoint, snapshotEndpoint,
		p.kvEventsConfig.TopicFilter, true); err != nil {
		logger.Error(err, "Failed to ensure KV-events subscriber for endpoint",
			"endpoint", endpointKey, "address", meta.Address)
		return fmt.Errorf("ensure subscriber for %s: %w", endpointKey, err)
	}

	logger.V(logging.DEBUG).Info("Ensured KV-events subscriber",
		"endpoint", endpointKey, "zmq", zmqEndpoint, "replay", replayEndpoint, "snapshot", snapshotEndpoint)
	return nil
}
