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

package metrics

// Attribute keys for the NIXL KV transfer failure counters an engine reports.
// Each value is the engine's cumulative count, so it restarts from zero when
// the engine restarts.
const (
	// NixlFailedTransfersKey holds the failed KV cache transfers.
	NixlFailedTransfersKey = "NixlFailedTransfers"
	// NixlFailedNotificationsKey holds the failed KV cache transfer notifications.
	NixlFailedNotificationsKey = "NixlFailedNotifications"
	// NixlKVExpiredRequestsKey holds the requests whose KV cache expired before it was read.
	NixlKVExpiredRequestsKey = "NixlKVExpiredRequests"
)

var (
	NixlFailedTransfersDataKey     = ScalarMetricDataKey(NixlFailedTransfersKey)
	NixlFailedNotificationsDataKey = ScalarMetricDataKey(NixlFailedNotificationsKey)
	NixlKVExpiredRequestsDataKey   = ScalarMetricDataKey(NixlKVExpiredRequestsKey)
)
