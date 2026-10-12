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

// nixlPushIdentity is the part of a NIXL push prefill answer's
// kv_transfer_params that is the same for every request to that prefill
// endpoint. Numbers stay float64, as decoded from JSON.
type nixlPushIdentity struct {
	engineID string
	host     string
	port     float64
	tpSize   float64
	// ppSize and dcpSize are 0 when the prefill answer leaves them out.
	ppSize  float64
	dcpSize float64
}

// kvTransferParams returns the identity as the kv_transfer_params fields of a
// decode request. It leaves out transfer_mode, which decode does not read, and
// the sizes the prefill answer left out.
func (id nixlPushIdentity) kvTransferParams() map[string]any {
	kv := map[string]any{
		reqcommon.FieldRemoteEngineID: id.engineID,
		reqcommon.FieldRemoteHost:     id.host,
		reqcommon.FieldRemotePort:     id.port,
		requestFieldTPSize:            id.tpSize,
	}
	if id.ppSize != 0 {
		kv[requestFieldPPSize] = id.ppSize
	}
	if id.dcpSize != 0 {
		kv[requestFieldDCPSize] = id.dcpSize
	}
	return kv
}

// extractNIXLPushIdentity returns the identity in a prefill answer's
// kv_transfer_params. It reports false unless transfer_mode is push and every
// identity field has its type; pp_size and dcp_size may be absent.
func extractNIXLPushIdentity(kvTransferParams any) (nixlPushIdentity, bool) {
	kv, ok := kvTransferParams.(map[string]any)
	if !ok || kv[requestFieldTransferMode] != nixlTransferModePush {
		return nixlPushIdentity{}, false
	}
	var identity nixlPushIdentity
	if identity.engineID, ok = kv[reqcommon.FieldRemoteEngineID].(string); !ok {
		return nixlPushIdentity{}, false
	}
	if identity.host, ok = kv[reqcommon.FieldRemoteHost].(string); !ok {
		return nixlPushIdentity{}, false
	}
	if identity.port, ok = kv[reqcommon.FieldRemotePort].(float64); !ok {
		return nixlPushIdentity{}, false
	}
	if identity.tpSize, ok = kv[requestFieldTPSize].(float64); !ok {
		return nixlPushIdentity{}, false
	}
	if identity.ppSize, ok = optionalNumber(kv, requestFieldPPSize); !ok {
		return nixlPushIdentity{}, false
	}
	if identity.dcpSize, ok = optionalNumber(kv, requestFieldDCPSize); !ok {
		return nixlPushIdentity{}, false
	}
	return identity, true
}

// optionalNumber returns kv[field] as a number, or 0 when kv has no such
// field. It reports false when the field holds anything but a number.
func optionalNumber(kv map[string]any, field string) (float64, bool) {
	value, present := kv[field]
	if !present {
		return 0, true
	}
	number, ok := value.(float64)
	return number, ok
}

// nixlPushIdentityCache maps a prefill endpoint (host:port) to the NIXL push
// identity it last answered with, so a later dispatch to it can build the
// decode request before the prefill answers. Safe for concurrent use.
type nixlPushIdentityCache struct {
	mu  sync.Mutex
	lru *simplelru.LRU[string, nixlPushIdentity]
	// noIdentityWarning warns once about a prefill answer without an identity.
	noIdentityWarning sync.Once
}

// get returns the identity cached for hostPort.
func (c *nixlPushIdentityCache) get(hostPort string) (nixlPushIdentity, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lru.Get(hostPort)
}

// put caches identity for hostPort, replacing the older entry.
func (c *nixlPushIdentityCache) put(hostPort string, identity nixlPushIdentity) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.lru.Add(hostPort, identity)
}

// remove removes the entry for hostPort. It reports whether there was one.
func (c *nixlPushIdentityCache) remove(hostPort string) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lru.Remove(hostPort)
}

// dropIfMatches removes the entry for hostPort only while it holds identity,
// so a failure seen with an older identity cannot remove a newer one. It
// reports whether it removed the entry.
func (c *nixlPushIdentityCache) dropIfMatches(hostPort string, identity nixlPushIdentity) bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	cached, ok := c.lru.Peek(hostPort)
	if !ok || cached != identity {
		return false
	}
	return c.lru.Remove(hostPort)
}

func newNIXLPushIdentityCache(size int) (*nixlPushIdentityCache, error) {
	lru, err := simplelru.NewLRU[string, nixlPushIdentity](size, nil)
	if err != nil {
		return nil, err
	}
	return &nixlPushIdentityCache{lru: lru}, nil
}

// storeNIXLPushIdentity caches the NIXL push identity in a prefill answer's
// kv_transfer_params under the endpoint that answered and returns it. An
// answer without one removes the endpoint's entry.
func (s *Server) storeNIXLPushIdentity(prefillPodHostPort string, kvTransferParams any) (nixlPushIdentity, bool) {
	identity, ok := extractNIXLPushIdentity(kvTransferParams)
	if !ok {
		s.nixlPushIdentities.remove(prefillPodHostPort)
		s.logMissingNIXLPushIdentity(prefillPodHostPort, kvTransferParams)
		return nixlPushIdentity{}, false
	}
	s.nixlPushIdentities.put(prefillPodHostPort, identity)
	s.logger.V(logging.TRACE).Info("stored NIXL push identity", "target", prefillPodHostPort,
		"engine_id", identity.engineID, "remote_host", identity.host, "remote_port", identity.port)
	return identity, true
}

// nixlPushIdentityMatches refreshes the cached identity from the prefill
// response and reports whether the prefill ran on the engine decode was sent to.
func (s *Server) nixlPushIdentityMatches(prefillPodHostPort string, sent nixlPushIdentity, prefillerResponse map[string]any) bool {
	answered, ok := s.storeNIXLPushIdentity(prefillPodHostPort, prefillerResponse[reqcommon.FieldKVTransferParams])
	return ok && answered == sent
}

// logMissingNIXLPushIdentity logs a prefill answer without a NIXL push
// identity. In NIXL push mode that points at a prefill pod whose connector or
// vLLM version does not match, so the first one is also a warning.
func (s *Server) logMissingNIXLPushIdentity(prefillPodHostPort string, kvTransferParams any) {
	var transferMode any
	if kv, ok := kvTransferParams.(map[string]any); ok {
		transferMode = kv[requestFieldTransferMode]
	}
	s.logger.V(logging.TRACE).Info("prefill answer carries no NIXL push identity",
		"target", prefillPodHostPort, "transfer_mode", transferMode)
	s.nixlPushIdentities.noIdentityWarning.Do(func() {
		s.logger.Info("warning: NIXL push mode is enabled but a prefill answer carries no NIXL push identity, "+
			"so its requests run serially. Run NixlPushConnector on vLLM v0.28.0 or newer on the prefill pods to fix this.",
			"target", prefillPodHostPort, "transfer_mode", transferMode)
	})
}
