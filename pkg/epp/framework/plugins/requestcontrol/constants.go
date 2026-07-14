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

// Package requestcontrol holds definitions shared by all requestcontrol framework plugins.
package requestcontrol

// TracerScope is the OTel instrumentation scope shared by all plugins under
// the requestcontrol framework component.
const TracerScope = "llm-d-router/pkg/epp/framework/plugins/requestcontrol"
