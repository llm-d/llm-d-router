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

package datalayer

import (
	"context"
	"encoding/json"
	"os"
	"sync"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const LocalStateStoreType = "local-state-store"

var _ fwkdl.CrossReplicaStore = (*LocalStateStore)(nil)

// LocalStateStore is an in-memory mock CrossReplicaStore for single-replica
// deployments and testing. No cross-replica sync is performed.
type LocalStateStore struct {
	typedName fwkplugin.TypedName
	replicaID string
	data      sync.Map // key: "replicaID:StateKey:endpointID", value: any
}

func NewLocalStateStore(name, replicaID string) *LocalStateStore {
	return &LocalStateStore{
		typedName: fwkplugin.TypedName{Type: LocalStateStoreType, Name: name},
		replicaID: replicaID,
	}
}

func LocalStateStoreFactory(name string, _ *json.Decoder, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	hostname, _ := os.Hostname()
	if hostname == "" {
		hostname = "local"
	}
	return NewLocalStateStore(name, hostname), nil
}

func (s *LocalStateStore) TypedName() fwkplugin.TypedName {
	return s.typedName
}

func (s *LocalStateStore) storeKey(key fwkdl.StateKey, endpointID string) string {
	return s.replicaID + ":" + string(key) + ":" + endpointID
}

func (s *LocalStateStore) Set(_ context.Context, key fwkdl.StateKey, endpointID string, value any) error {
	s.data.Store(s.storeKey(key, endpointID), value)
	return nil
}

func (s *LocalStateStore) Get(_ context.Context, key fwkdl.StateKey, endpointID string) (any, bool, error) {
	v, ok := s.data.Load(s.storeKey(key, endpointID))
	return v, ok, nil
}

func (s *LocalStateStore) Delete(_ context.Context, key fwkdl.StateKey, endpointID string) error {
	s.data.Delete(s.storeKey(key, endpointID))
	return nil
}
