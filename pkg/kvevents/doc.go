// Copyright 2025 The llm-d Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build !ignore

// Package kvevents contains the implementation of the KV-events processing
// system. It provides a pool for handling events coming from a distributed
// KV-cache pool, allowing for cache-tracking and real-time updates
// to the KV-cache index. The package is designed to work with the
// kvcache.Indexer to maintain an up-to-date state of the KV-cache
// and to facilitate the scoring of pods based on the KV-cache index state.
package kvevents
