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

package models

import (
	"fmt"
	"strings"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const (
	ModelsExtractorType = "models-data-extractor"
)

var ModelsAttributeKey = plugin.NewDataKey("/v1/models", ModelsExtractorType)

// ModelDataCollection defines models' data returned from /v1/models API
type ModelDataCollection []ModelData

// ModelData defines model's data returned from /v1/models API
type ModelData struct {
	ID     string `json:"id"`
	Parent string `json:"parent,omitempty"`
}

// String returns a string representation of the model info
func (m *ModelData) String() string {
	return fmt.Sprintf("%+v", *m)
}

// Clone returns a full copy of the object
func (m ModelDataCollection) Clone() fwkdl.Cloneable {
	if m == nil {
		return nil
	}

	clone := make(ModelDataCollection, len(m))
	copy(clone, m)
	return clone
}

func (m ModelDataCollection) String() string {
	if m == nil {
		return "[]"
	}
	parts := make([]string, len(m))
	for i, p := range m {
		parts[i] = p.String()
	}
	return "[" + strings.Join(parts, ", ") + "]"
}
