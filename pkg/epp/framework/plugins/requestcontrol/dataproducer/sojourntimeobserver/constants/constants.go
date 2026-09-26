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

package constants

// SojournTimeObserverProducerType is the plugin type of the producer that
// observes per-endpoint sojourn samples, split into TTFT (dispatchedAt to
// firstChunkAt) and decode (firstChunkAt to endOfStreamAt), and publishes
// the paired t-digest snapshot the mrl-scorer-hub reads.
const SojournTimeObserverProducerType = "sojourn-time-observer-hub"
