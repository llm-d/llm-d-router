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

package local

import (
	"context"
	"encoding/json"
	"os"
	"sync"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const LocalSyncerType = "local-syncer"

var _ fwkdl.CrossReplicaSyncer = (*LocalSyncer)(nil)

// LocalSyncer is an in-memory CrossReplicaSyncer for single-replica
// deployments and testing. No cross-replica synchronization is performed.
type LocalSyncer struct {
	typedName fwkplugin.TypedName
	replicaID string
	data      sync.Map
}

func NewLocalSyncer(name, replicaID string) *LocalSyncer {
	return &LocalSyncer{
		typedName: fwkplugin.TypedName{Type: LocalSyncerType, Name: name},
		replicaID: replicaID,
	}
}

func LocalSyncerFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	hostname, _ := os.Hostname()
	if hostname == "" {
		hostname = "local"
	}
	return NewLocalSyncer(name, hostname), nil
}

func (s *LocalSyncer) TypedName() fwkplugin.TypedName {
	return s.typedName
}

func (s *LocalSyncer) syncKey(key fwkdl.StateKey, id string) string {
	return s.replicaID + ":" + string(key) + ":" + id
}

func (s *LocalSyncer) Set(_ context.Context, key fwkdl.StateKey, endpointID string, value any, aggregate func([]any) any) error {
	s.data.Store(s.syncKey(key, endpointID), aggregate([]any{value}))
	return nil
}

func (s *LocalSyncer) Get(_ context.Context, key fwkdl.StateKey, endpointID string) (any, bool, error) {
	value, ok := s.data.Load(s.syncKey(key, endpointID))
	return value, ok, nil
}

func (s *LocalSyncer) Delete(_ context.Context, key fwkdl.StateKey, endpointID string) error {
	s.data.Delete(s.syncKey(key, endpointID))
	return nil
}

func (s *LocalSyncer) GetOrSet(_ context.Context, key fwkdl.StateKey, id string, candidate any) (any, bool, error) {
	actual, loaded := s.data.LoadOrStore(s.syncKey(key, id), candidate)
	return actual, loaded, nil
}
