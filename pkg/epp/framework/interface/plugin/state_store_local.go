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

import "sync"

// LocalStateStore is an in-memory StateStore backed by a sync.Map.
type LocalStateStore struct {
	data sync.Map
}

func NewLocalStateStore() *LocalStateStore {
	return &LocalStateStore{}
}

func (s *LocalStateStore) Get(key StateKey, id string) (any, bool) {
	return s.data.Load(string(key) + ":" + id)
}

func (s *LocalStateStore) Set(key StateKey, id string, value any) {
	s.data.Store(string(key)+":"+id, value)
}

func (s *LocalStateStore) Delete(key StateKey, id string) {
	s.data.Delete(string(key) + ":" + id)
}
