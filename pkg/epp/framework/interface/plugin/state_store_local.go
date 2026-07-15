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

package plugin

import (
	"encoding/json"
	"os"
	"sync"
)

const LocalStateStoreType = "local-state-store"

// ReplicaID uniquely identifies one EPP replica.
type ReplicaID string

// replicaID returns a ReplicaID from the POD_NAME environment variable,
// falling back to "local" when unset.
func replicaID() ReplicaID {
	if name := os.Getenv("POD_NAME"); name != "" {
		return ReplicaID(name)
	}
	return "local"
}

var _ SharedStateStore = (*LocalStateStore)(nil)

// LocalStateStore is an in-memory SharedStateStore backed by a sync.Map.
type LocalStateStore struct {
	typedName TypedName
	self      ReplicaID
	data      sync.Map
}

func NewLocalStateStore(name string, self ReplicaID) *LocalStateStore {
	return &LocalStateStore{
		typedName: TypedName{Type: LocalStateStoreType, Name: name},
		self:      self,
	}
}

// LocalStateStoreFactory creates a LocalStateStore from plugin config.
func LocalStateStoreFactory(name string, _ *json.Decoder, _ Handle) (Plugin, error) {
	return NewLocalStateStore(name, replicaID()), nil
}

func (s *LocalStateStore) TypedName() TypedName {
	return s.typedName
}

func (s *LocalStateStore) Set(key StateKey, id string, value any) {
	s.data.Store(string(key)+":"+id, func() any { return value })
}

func (s *LocalStateStore) Publish(key StateKey, id string, supplier func() any) {
	s.data.Store(string(key)+":"+id, supplier)
}

func (s *LocalStateStore) Get(key StateKey, id string) (any, bool) {
	v, ok := s.data.Load(string(key) + ":" + id)
	if !ok {
		return nil, false
	}
	return v.(func() any)(), true
}

func (s *LocalStateStore) Delete(key StateKey, id string) {
	s.data.Delete(string(key) + ":" + id)
}
