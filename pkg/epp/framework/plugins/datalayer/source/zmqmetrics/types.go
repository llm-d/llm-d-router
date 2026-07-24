/*
Copyright 2026 The Kubernetes Authors.

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

package zmqmetrics

// ZmqMetricsStats represents the Msgpack serialization schema
// sent by the model server's ZMQ publisher.
type ZmqMetricsStats struct {
	NumRequestsRunning int            `msgpack:"num_requests_running"`
	NumRequestsWaiting int            `msgpack:"num_requests_waiting"`
	KVCacheUsagePerc   float64        `msgpack:"kv_cache_usage_perc"`
	CacheConfigInfo    map[string]any `msgpack:"cache_config_info"`
	EngineID           string         `msgpack:"engine_id"`
}
