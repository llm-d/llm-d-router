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
	"fmt"
	"net"
	"strconv"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/sglang/gen/apiv1"
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
		return 0, false, fmt.Errorf("server info without deployment dp rank")
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
