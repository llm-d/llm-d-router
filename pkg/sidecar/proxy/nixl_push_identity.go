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

package proxy

import (
	"maps"
	"sync"

	"github.com/hashicorp/golang-lru/v2/simplelru"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// NIXL push kv_transfer_params fields and values.
const (
	requestFieldTPSize       = "tp_size"
	requestFieldPPSize       = "pp_size"
	requestFieldDCPSize      = "dcp_size"
	requestFieldTransferMode = "transfer_mode"

	nixlTransferModePush = "push"
)

// nixlPushIdentityFields maps each identity field to whether a prefill answer
// must carry it to be cached. vLLM defaults the optional sizes to 1.
var nixlPushIdentityFields = map[string]bool{
	reqcommon.FieldRemoteEngineID: true,
	reqcommon.FieldRemoteHost:     true,
	reqcommon.FieldRemotePort:     true,
	requestFieldTPSize:            true,
	requestFieldTransferMode:      true,
	requestFieldPPSize:            false,
	requestFieldDCPSize:           false,
}

// nixlPushIdentity is the part of a NIXL push prefill answer's
// kv_transfer_params that is the same for every request to that prefill
// endpoint. Values stay as decoded so they reach a decode request unchanged.
type nixlPushIdentity map[string]any

func (id nixlPushIdentity) equal(other nixlPushIdentity) bool {
	return maps.Equal(id, other)
}

// extractNIXLPushIdentity returns the identity in a prefill answer's
// kv_transfer_params. It reports false unless transfer_mode is push and every
// required field is present. Identity fields must be JSON strings or numbers
// so identities compare and copy as plain values.
func extractNIXLPushIdentity(kvTransferParams any) (nixlPushIdentity, bool) {
	kv, ok := kvTransferParams.(map[string]any)
	if !ok || kv[requestFieldTransferMode] != nixlTransferModePush {
		return nil, false
	}
	identity := make(nixlPushIdentity, len(nixlPushIdentityFields))
	for field, required := range nixlPushIdentityFields {
		value, present := kv[field]
		if !present && !required {
			continue
		}
		switch value.(type) {
		case string, float64:
			identity[field] = value
		default:
			return nil, false
		}
	}
	return identity, true
}

// nixlPushIdentityCache maps a prefill endpoint (host:port) to the NIXL push
// identity it last answered with, so a later dispatch to it can build the
// decode request before the prefill answers. Safe for concurrent use.
type nixlPushIdentityCache struct {
	mu  sync.Mutex
	lru *simplelru.LRU[string, nixlPushIdentity]
}

func newNIXLPushIdentityCache(size int) (*nixlPushIdentityCache, error) {
	lru, err := simplelru.NewLRU[string, nixlPushIdentity](size, nil)
	if err != nil {
		return nil, err
	}
	return &nixlPushIdentityCache{lru: lru}, nil
}

// get returns a copy of the identity cached for hostPort, which the caller may
// modify.
func (c *nixlPushIdentityCache) get(hostPort string) (nixlPushIdentity, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	identity, ok := c.lru.Get(hostPort)
	if !ok {
		return nil, false
	}
	return maps.Clone(identity), true
}

// put caches a copy of identity for hostPort, replacing the older entry.
func (c *nixlPushIdentityCache) put(hostPort string, identity nixlPushIdentity) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.lru.Add(hostPort, maps.Clone(identity))
}

// dropIfMatches removes the entry for hostPort only while it holds identity,
// so a failure seen with an older identity cannot remove a newer one. It
// reports whether it removed the entry.
func (c *nixlPushIdentityCache) dropIfMatches(hostPort string, identity nixlPushIdentity) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	cached, ok := c.lru.Peek(hostPort)
	if !ok || !cached.equal(identity) {
		return false
	}
	return c.lru.Remove(hostPort)
}

// storeNIXLPushIdentity caches the NIXL push identity in a prefill answer's
// kv_transfer_params under the endpoint that answered. An answer without one
// leaves the cache unchanged.
func (s *Server) storeNIXLPushIdentity(prefillPodHostPort string, kvTransferParams any) {
	identity, ok := extractNIXLPushIdentity(kvTransferParams)
	if !ok {
		return
	}
	s.nixlPushIdentities.put(prefillPodHostPort, identity)
	s.logger.V(logging.TRACE).Info("stored NIXL push identity", "target", prefillPodHostPort, "identity", identity)
}
