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

package pipeline

// TracerScope is the OTel instrumentation scope for spans emitted by the
// coordinator pipeline.
const TracerScope = "llm-d-router/pkg/coordinator/pipeline"

// pipelineSpanName is the span covering one run of the step list.
const pipelineSpanName = "pipeline"
