/*
Copyright 2026 The Kubernetes Authors.

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

package statestore

import (
	"context"

	pb "github.com/llm-d/llm-d-router/pkg/epp/statestore/stateapi/proto/gen"
)

// remoteInflightState implements InflightState over the gRPC State API. It is
// never used unwrapped in production wiring: callers get it decorated by
// NewFailOpenInflightState so a stateful-EPP outage degrades to the Local
// provider rather than failing requests.
type remoteInflightState struct {
	client pb.StateAPIClient
}

// NewRemoteInflightState returns an InflightState backed by the given gRPC
// State API client.
func NewRemoteInflightState(client pb.StateAPIClient) InflightState {
	return &remoteInflightState{client: client}
}

func (r *remoteInflightState) GetInflightSnapshot(ctx context.Context, endpointID string) InflightSnapshot {
	batch := r.GetInflightSnapshotBatch(ctx, []string{endpointID})
	return batch[endpointID]
}

func (r *remoteInflightState) GetInflightSnapshotBatch(ctx context.Context, endpointIDs []string) map[string]InflightSnapshot {
	resp, err := r.client.GetInflightSnapshotBatch(ctx, &pb.InflightSnapshotBatchRequest{EndpointIds: endpointIDs})
	if err != nil {
		return nil
	}
	result := make(map[string]InflightSnapshot, len(resp.GetSnapshots()))
	for id, snap := range resp.GetSnapshots() {
		result[id] = InflightSnapshot{Requests: snap.GetRequests(), Tokens: snap.GetTokens()}
	}
	return result
}

func (r *remoteInflightState) ReserveInflight(ctx context.Context, requestID, endpointID string, estimatedTokens int64) error {
	_, err := r.client.ReserveInflight(ctx, &pb.ReserveInflightRequest{
		RequestId: requestID, EndpointId: endpointID, EstimatedTokens: estimatedTokens,
	})
	return err
}

func (r *remoteInflightState) ReleaseInflight(ctx context.Context, requestID, endpointID string, estimatedTokens int64) error {
	_, err := r.client.ReleaseInflight(ctx, &pb.ReleaseInflightRequest{
		RequestId: requestID, EndpointId: endpointID, EstimatedTokens: estimatedTokens,
	})
	return err
}

func (r *remoteInflightState) DeleteEndpoint(ctx context.Context, endpointID string) {
	_, _ = r.client.DeleteInflightEndpoint(ctx, &pb.DeleteEndpointRequest{EndpointId: endpointID})
}

// remotePrefixState implements PrefixState over the gRPC State API. Like
// remoteInflightState, it is only ever used wrapped by
// NewFailOpenPrefixState.
type remotePrefixState struct {
	client pb.StateAPIClient
}

// NewRemotePrefixState returns a PrefixState backed by the given gRPC State
// API client.
func NewRemotePrefixState(client pb.StateAPIClient) PrefixState {
	return &remotePrefixState{client: client}
}

func (r *remotePrefixState) GetPrefixMatch(ctx context.Context, hash uint64) []string {
	batch := r.GetPrefixMatchBatch(ctx, []uint64{hash})
	return batch[hash]
}

func (r *remotePrefixState) GetPrefixMatchBatch(ctx context.Context, hashes []uint64) map[uint64][]string {
	resp, err := r.client.GetPrefixMatchBatch(ctx, &pb.PrefixMatchBatchRequest{Hashes: hashes})
	if err != nil {
		return nil
	}
	result := make(map[uint64][]string, len(resp.GetMatches()))
	for hash, endpoints := range resp.GetMatches() {
		result[hash] = endpoints.GetEndpointIds()
	}
	return result
}

func (r *remotePrefixState) CommitPrefix(ctx context.Context, requestID, endpointID string, hashes []uint64) error {
	_, err := r.client.CommitPrefix(ctx, &pb.CommitPrefixRequest{RequestId: requestID, EndpointId: endpointID, Hashes: hashes})
	return err
}

func (r *remotePrefixState) RemoveEndpoint(ctx context.Context, endpointID string) {
	_, _ = r.client.RemovePrefixEndpoint(ctx, &pb.DeleteEndpointRequest{EndpointId: endpointID})
}
