/*
Copyright 2025 The Kubernetes Authors.

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
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"
)

// fakeStore records calls to BackendUpsert and BackendDelete for assertion.
type fakeStore struct {
	upserted []*EndpointMetadata
	deleted  []types.NamespacedName
}

func (f *fakeStore) BackendUpsert(_ context.Context, meta *EndpointMetadata) {
	f.upserted = append(f.upserted, meta)
}

func (f *fakeStore) BackendDelete(id types.NamespacedName) {
	f.deleted = append(f.deleted, id)
}

func TestNewDiscoveryNotifier_Upsert(t *testing.T) {
	store := &fakeStore{}
	notifier := NewDiscoveryNotifier(store)
	meta := &EndpointMetadata{
		NamespacedName: types.NamespacedName{Name: "ep1", Namespace: "default"},
		Address:        "10.0.0.1",
	}

	notifier.Upsert(meta)

	assert.Len(t, store.upserted, 1)
	assert.Equal(t, meta, store.upserted[0])
	assert.Empty(t, store.deleted)
}

func TestNewDiscoveryNotifier_Delete(t *testing.T) {
	store := &fakeStore{}
	notifier := NewDiscoveryNotifier(store)
	id := types.NamespacedName{Name: "ep1", Namespace: "default"}

	notifier.Delete(id)

	assert.Empty(t, store.upserted)
	assert.Len(t, store.deleted, 1)
	assert.Equal(t, id, store.deleted[0])
}

func TestNewDiscoveryNotifier_UpsertThenDelete(t *testing.T) {
	store := &fakeStore{}
	notifier := NewDiscoveryNotifier(store)
	id := types.NamespacedName{Name: "ep1", Namespace: "default"}

	notifier.Upsert(&EndpointMetadata{NamespacedName: id})
	notifier.Delete(id)

	assert.Len(t, store.upserted, 1)
	assert.Len(t, store.deleted, 1)
}
