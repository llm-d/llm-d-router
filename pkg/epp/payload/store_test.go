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

package payload

import (
	"context"
	"errors"
	"testing"
)

func TestNoopStoreDiscards(t *testing.T) {
	uri, err := NoopStore{}.Store(context.Background(), PayloadRef{}, []byte("anything"))
	if err != nil {
		t.Fatalf("NoopStore.Store returned error: %v", err)
	}
	if uri != "" {
		t.Fatalf("NoopStore.Store returned URI %q, want empty", uri)
	}
}

func TestInlineStoreThreshold(t *testing.T) {
	store := InlineStore{MaxBytes: 8}

	uri, err := store.Store(context.Background(), PayloadRef{}, []byte("12345678"))
	if err != nil {
		t.Fatalf("Store at threshold returned error: %v", err)
	}
	if uri != "" {
		t.Fatalf("InlineStore.Store returned URI %q, want empty", uri)
	}

	_, err = store.Store(context.Background(), PayloadRef{}, []byte("123456789"))
	if !errors.Is(err, ErrPayloadTooLarge) {
		t.Fatalf("Store above threshold returned %v, want ErrPayloadTooLarge", err)
	}
}
