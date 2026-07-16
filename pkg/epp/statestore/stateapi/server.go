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

package stateapi

import (
	"context"

	pb "github.com/llm-d/llm-d-router/pkg/epp/statestore/stateapi/proto/gen"
)

// Server implements the generated StateAPIServer gRPC interface by
// delegating to a Store. Bare gRPC (no TLS), matching the existing internal
// health-check server's precedent in this repo — acceptable for this
// feasibility spike, not for production.
type Server struct {
	pb.UnimplementedStateAPIServer
	store Store
}

var _ pb.StateAPIServer = (*Server)(nil)

// NewServer returns a Server backed by the given Store.
func NewServer(store Store) *Server {
	return &Server{store: store}
}

func (s *Server) GetInflightSnapshotBatch(_ context.Context, req *pb.InflightSnapshotBatchRequest) (*pb.InflightSnapshotBatchResponse, error) {
	snapshots := s.store.SnapshotBatch(req.GetEndpointIds())
	resp := &pb.InflightSnapshotBatchResponse{Snapshots: make(map[string]*pb.InflightSnapshot, len(snapshots))}
	for id, snap := range snapshots {
		resp.Snapshots[id] = &pb.InflightSnapshot{Requests: snap.Requests, Tokens: snap.Tokens}
	}
	return resp, nil
}

func (s *Server) ReserveInflight(_ context.Context, req *pb.ReserveInflightRequest) (*pb.ReserveInflightResponse, error) {
	s.store.Reserve(req.GetRequestId(), req.GetEndpointId(), req.GetEstimatedTokens())
	return &pb.ReserveInflightResponse{}, nil
}

func (s *Server) ReleaseInflight(_ context.Context, req *pb.ReleaseInflightRequest) (*pb.ReleaseInflightResponse, error) {
	s.store.Release(req.GetRequestId(), req.GetEndpointId(), req.GetEstimatedTokens())
	return &pb.ReleaseInflightResponse{}, nil
}

func (s *Server) DeleteInflightEndpoint(_ context.Context, req *pb.DeleteEndpointRequest) (*pb.DeleteEndpointResponse, error) {
	s.store.DeleteEndpoint(req.GetEndpointId())
	return &pb.DeleteEndpointResponse{}, nil
}

func (s *Server) GetPrefixMatchBatch(_ context.Context, req *pb.PrefixMatchBatchRequest) (*pb.PrefixMatchBatchResponse, error) {
	resp := &pb.PrefixMatchBatchResponse{Matches: make(map[uint64]*pb.PrefixMatchEndpoints, len(req.GetHashes()))}
	for _, hash := range req.GetHashes() {
		ids := s.store.Match(hash)
		if len(ids) == 0 {
			continue
		}
		resp.Matches[hash] = &pb.PrefixMatchEndpoints{EndpointIds: ids}
	}
	return resp, nil
}

func (s *Server) CommitPrefix(_ context.Context, req *pb.CommitPrefixRequest) (*pb.CommitPrefixResponse, error) {
	s.store.Commit(req.GetRequestId(), req.GetEndpointId(), req.GetHashes())
	return &pb.CommitPrefixResponse{}, nil
}

func (s *Server) RemovePrefixEndpoint(_ context.Context, req *pb.DeleteEndpointRequest) (*pb.DeleteEndpointResponse, error) {
	s.store.RemoveEndpoint(req.GetEndpointId())
	return &pb.DeleteEndpointResponse{}, nil
}

func (s *Server) AdmitConcurrency(_ context.Context, req *pb.AdmitConcurrencyRequest) (*pb.AdmitConcurrencyResponse, error) {
	outcome := s.store.Admit(req.GetRequestId(), flowKeyFromProto(req.GetFlowKey()))
	return &pb.AdmitConcurrencyResponse{Outcome: outcomeToProto(outcome)}, nil
}

func (s *Server) ReleaseConcurrency(_ context.Context, req *pb.ReleaseConcurrencyRequest) (*pb.ReleaseConcurrencyResponse, error) {
	s.store.ReleaseConcurrency(req.GetRequestId(), flowKeyFromProto(req.GetFlowKey()))
	return &pb.ReleaseConcurrencyResponse{}, nil
}

func flowKeyFromProto(k *pb.FlowKey) FlowKey {
	if k == nil {
		return FlowKey{}
	}
	return FlowKey{ID: k.GetId(), Priority: int(k.GetPriority())}
}

func outcomeToProto(o ConcurrencyOutcome) pb.ConcurrencyOutcome {
	if o == ConcurrencyOutcomeAdmitted {
		return pb.ConcurrencyOutcome_CONCURRENCY_OUTCOME_ADMITTED
	}
	return pb.ConcurrencyOutcome_CONCURRENCY_OUTCOME_REJECTED
}
