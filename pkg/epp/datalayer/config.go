/*
Copyright 2025 The Kubernetes Authors.

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

package datalayer

import (
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

// Config defines the configuration of EPP data layer, as the set of DataSources
// and Extractors defined on them. Both poll-based and event-driven (notification)
// sources are stored in Sources. Differentiation by type of source is handled during
// the set-up phase.
type Config struct {
	Sources []DataSourceConfig // the data sources configured in the data layer
}

// DataSourceConfig defines the configuration of a specific DataSource.
// Plugin may be a DataSource (notification, endpoint) or a PollingDispatcher;
// the framework type-asserts to the right variant at Configure time.
// Extractors are stored as plugin.Plugin and type-asserted to the source's
// variant; PollingDispatchers consume them via AppendExtractor.
type DataSourceConfig struct {
	Plugin     plugin.Plugin   // the source plugin instance (DataSource or PollingDispatcher)
	Extractors []plugin.Plugin // extractors defined for the data source
}
