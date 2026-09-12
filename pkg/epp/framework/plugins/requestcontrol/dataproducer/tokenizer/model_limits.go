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

package tokenizer

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/common/observability/logging"
)

const modelLimitFreshness = 90 * time.Second

type renderCapability struct {
	url       string
	declared  int
	observed  int
	refreshed time.Time
}

// eligible is called with the picker read lock held.
func (p *discoveredEndpointPicker) eligible(c *renderCapability, now time.Time) bool {
	limit := c.declared
	if p.config.DiscoverModelLimits {
		if c.observed <= 0 || now.Sub(c.refreshed) > modelLimitFreshness {
			return false
		}
		if limit == 0 || c.observed < limit {
			limit = c.observed
		}
	}
	return limit >= p.config.MinModelLen
}

// watchModelLimits keeps network I/O outside endpoint notification callbacks.
func (p *discoveredEndpointPicker) watchModelLimits(ctx context.Context, client *http.Client, model string) {
	ticker := time.NewTicker(30 * time.Second)
	defer ticker.Stop()
	for {
		p.refreshModelLimits(ctx, client, model)
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
		}
	}
}

// refreshModelLimits probes a bounded snapshot. Replaced endpoints cannot inherit
// results from probes of their predecessors.
func (p *discoveredEndpointPicker) refreshModelLimits(ctx context.Context, client *http.Client, model string) {
	p.mu.RLock()
	snapshot := make(map[string]*renderCapability, len(p.capabilities))
	for id, c := range p.capabilities {
		snapshot[id] = c
	}
	p.mu.RUnlock()
	semaphore := make(chan struct{}, 4)
	var wg sync.WaitGroup
	for id, c := range snapshot {
		select {
		case semaphore <- struct{}{}:
		case <-ctx.Done():
			wg.Wait()
			return
		}
		wg.Add(1)
		go func(id string, c *renderCapability) {
			defer wg.Done()
			defer func() { <-semaphore }()
			probeCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
			defer cancel()
			limit, err := readModelLimit(probeCtx, client, c.url, model)
			p.mu.Lock()
			changed := false
			if p.capabilities[id] == c {
				changed = c.observed != limit
				c.observed = limit
				c.refreshed = time.Now()
			}
			p.mu.Unlock()
			if err != nil {
				log.FromContext(ctx).Error(err, "renderer model discovery failed", "endpoint", id)
			}
			if changed && err == nil {
				log.FromContext(ctx).V(logging.DEBUG).Info("renderer model limit discovered", "endpoint", id, "maxModelLen", limit)
			}
		}(id, c)
	}
	wg.Wait()
}

func readModelLimit(ctx context.Context, client *http.Client, url, model string) (int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url+"/v1/models", nil)
	if err != nil {
		return 0, err
	}
	if auth := vllmWarmupAuthHeader(); auth != "" {
		req.Header.Set("Authorization", auth)
	}
	resp, err := client.Do(req)
	if err != nil {
		return 0, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("models endpoint returned HTTP %d", resp.StatusCode)
	}
	var body struct {
		Data []struct {
			ID          string `json:"id"`
			MaxModelLen int    `json:"max_model_len"`
		} `json:"data"`
	}
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&body); err != nil {
		return 0, err
	}
	for _, m := range body.Data {
		if m.ID == model && m.MaxModelLen > 0 {
			return m.MaxModelLen, nil
		}
	}
	return 0, fmt.Errorf("models endpoint has no positive max_model_len for %q", model)
}
