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

package kvblock

import "sync"

// CacheFormat describes the token matching condition and page geometry.
type CacheFormat struct {
	Bigram    bool
	BlockSize int
}

// CacheFormatCatalog holds per-pod formats learned from KV events.
type CacheFormatCatalog struct {
	mu      sync.RWMutex
	entries map[string]CacheFormat
}

// NewCacheFormatCatalog creates an empty catalog.
func NewCacheFormatCatalog() *CacheFormatCatalog {
	return &CacheFormatCatalog{entries: make(map[string]CacheFormat)}
}

// Learn records the format of a pod's cache.
func (c *CacheFormatCatalog) Learn(podID string, format CacheFormat) {
	if current, ok := c.Get(podID); ok && current == format {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.entries == nil {
		c.entries = make(map[string]CacheFormat)
	}
	c.entries[podID] = format
}

// Get returns a pod's learned format.
func (c *CacheFormatCatalog) Get(podID string) (CacheFormat, bool) {
	if c == nil {
		return CacheFormat{}, false
	}
	c.mu.RLock()
	defer c.mu.RUnlock()
	format, ok := c.entries[podID]
	return format, ok
}

// Clear forgets the format when a pod retires.
func (c *CacheFormatCatalog) Clear(podID string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	delete(c.entries, podID)
}
