/*
Copyright 2026 The Kubernetes Authors.
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

package approximateprefix

import (
	"context"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// backgroundMergeBatchPrompts bounds Add calls before a waiting Extract runs.
const backgroundMergeBatchPrompts = 32

// registeredEndpoint serializes lifecycle writes with this endpoint's Add batches.
type registeredEndpoint struct {
	mu       sync.RWMutex
	endpoint datalayer.Endpoint
}

type mergeStats struct {
	promptsAdded  int
	digestSkipped int
}

// registeredMergeEndpoint shares the datalayer attributes without cloning them.
// Endpoint retains the exact registration identity for the check before Add.
type registeredMergeEndpoint struct {
	datalayer.Endpoint
	datalayer.AttributeMap
	registration *registeredEndpoint
}

var _ fwksched.Endpoint = registeredMergeEndpoint{}

func (e registeredMergeEndpoint) Peek(key plugin.DataKey) (datalayer.Cloneable, bool) {
	type peeker interface {
		Peek(plugin.DataKey) (datalayer.Cloneable, bool)
	}
	if attributes, ok := e.AttributeMap.(peeker); ok {
		return attributes.Peek(key)
	}
	return e.Get(key)
}

// lockEndpointRegistration retries when a delete retires the lock being acquired.
func (p *dataProducer) lockEndpointRegistration(endpointID string) *registeredEndpoint {
	for {
		value, ok := p.registeredEndpoints.Load(endpointID)
		if !ok {
			value, _ = p.registeredEndpoints.LoadOrStore(endpointID, &registeredEndpoint{})
		}
		registration := value.(*registeredEndpoint)
		registration.mu.Lock()
		if current, ok := p.registeredEndpoints.Load(endpointID); ok && current == registration {
			return registration
		}
		registration.mu.Unlock()
	}
}

func (e registeredMergeEndpoint) lockCurrent(p *dataProducer, endpointID string) bool {
	e.registration.mu.RLock()
	current, ok := p.registeredEndpoints.Load(endpointID)
	if !ok || current != e.registration || e.registration.endpoint != e.Endpoint {
		e.registration.mu.RUnlock()
		return false
	}
	return true
}

func (p *dataProducer) registeredMergeEndpoints() []registeredMergeEndpoint {
	var endpoints []registeredMergeEndpoint
	p.registeredEndpoints.Range(func(_, value any) bool {
		registration := value.(*registeredEndpoint)
		registration.mu.RLock()
		if ep := registration.endpoint; ep != nil {
			endpoints = append(endpoints, registeredMergeEndpoint{Endpoint: ep, AttributeMap: ep.GetAttributes(), registration: registration})
		}
		registration.mu.RUnlock()
		return true
	})
	return endpoints
}

// AfterCrossReplicaPublish queues a coalesced local-cache read and returns
// without waiting for the worker. Its lifetime is the plugin's constructor ctx.
func (p *dataProducer) AfterCrossReplicaPublish(ctx context.Context) {
	if p.config.BackgroundMerge.Trigger != backgroundMergeCallback || ctx.Err() != nil ||
		p.backgroundMergeCtx.Err() != nil || !p.syncerActive() {
		return
	}
	select {
	case p.backgroundMergeSignals <- struct{}{}:
	default:
	}
}

func (p *dataProducer) startBackgroundMerge() {
	if p.config.BackgroundMerge.Trigger != backgroundMergeCallback || !p.syncerActive() || p.backgroundMergeCtx.Err() != nil {
		return
	}
	p.backgroundMergeOnce.Do(func() {
		p.backgroundMergeDone = make(chan struct{})
		go p.runBackgroundMerge()
	})
}

func (p *dataProducer) runBackgroundMerge() {
	defer close(p.backgroundMergeDone)
	for {
		if p.backgroundMergeCtx.Err() != nil {
			return
		}
		select {
		case <-p.backgroundMergeCtx.Done():
			return
		case <-p.backgroundMergeSignals:
		}
		p.mergeInBackground(p.backgroundMergeCtx)
	}
}

func (p *dataProducer) mergeInBackground(ctx context.Context) {
	if ctx.Err() != nil || !p.syncerActive() {
		return
	}
	started := time.Now()
	endpoints := p.registeredMergeEndpoints()
	stats := p.mergeRemoteHashesInBackground(endpoints)
	log.FromContext(ctx).V(logutil.DEBUG).Info("Background prefix merge",
		"endpoints", len(endpoints), "promptsAdded", stats.promptsAdded, "digestSkipped", stats.digestSkipped, "duration", time.Since(started))
}

// mergeRemoteHashesInBackground checks registration identity before each Add batch.
func (p *dataProducer) mergeRemoteHashesInBackground(endpoints []registeredMergeEndpoint) mergeStats {
	var stats mergeStats
	if !p.syncerActive() || p.indexerInst == nil {
		return stats
	}
endpoints:
	for _, snapshot := range endpoints {
		if p.backgroundMergeCtx.Err() != nil {
			break
		}
		set, endpointID, digest := p.peerPrefixHashSet(snapshot)
		if set == nil {
			continue
		}
		if p.peerMergedDigestMatches(endpointID, digest) {
			stats.digestSkipped++
			continue
		}
		active := p.activePeerKeys(endpointID, set.Prompts)
		novel := p.unmergedPeerKeys(endpointID, active)
		pending := make([][]uint64, 0, len(novel))
		for _, prompt := range set.Prompts {
			if len(prompt) > 0 {
				if _, ok := novel[promptKeyUint64(prompt)]; ok {
					pending = append(pending, prompt)
				}
			}
		}
		if p.backgroundMergeCtx.Err() != nil || !snapshot.lockCurrent(p, endpointID) {
			continue
		}
		// PodList removal precedes Extract(delete); cleanup covers that writeback
		// window, unless the same name returns before cleanup observes its absence.
		server := p.makeserver(snapshot)
		batch := 0
		for _, prompt := range pending {
			p.indexerInst.Add(uint64sToBlockHashes(prompt), server)
			stats.promptsAdded++
			batch++
			if batch == backgroundMergeBatchPrompts {
				snapshot.registration.mu.RUnlock()
				batch = 0
				// A waiting Extract writer gets the lock before the next read batch.
				if p.backgroundMergeCtx.Err() != nil || !snapshot.lockCurrent(p, endpointID) {
					continue endpoints
				}
			}
		}
		// Commit only a completed window; interrupted batches must be retried.
		p.rememberPeerMergedWindow(endpointID, active, digest)
		snapshot.registration.mu.RUnlock()
	}
	return stats
}

// unmergedPeerKeys prepares the batch outside the registration lock.
func (p *dataProducer) unmergedPeerKeys(endpointID string, active map[string]struct{}) map[string]struct{} {
	value, ok := p.peerMerged.Load(endpointID)
	if !ok {
		return active
	}
	remembered := value.(*mergedKeySet)
	remembered.mu.Lock()
	defer remembered.mu.Unlock()
	novel := make(map[string]struct{})
	for key := range active {
		if _, ok := remembered.keys[key]; !ok {
			novel[key] = struct{}{}
		}
	}
	return novel
}

func (p *dataProducer) rememberPeerMergedWindow(endpointID string, active map[string]struct{}, digest uint64) {
	value, _ := p.peerMerged.LoadOrStore(endpointID, &mergedKeySet{})
	remembered := value.(*mergedKeySet)
	remembered.mu.Lock()
	remembered.keys = active
	remembered.digest = digest
	remembered.haveDigest = true
	remembered.mu.Unlock()
}
