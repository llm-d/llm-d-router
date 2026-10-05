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

package models

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
)

const (
	ModelsExtractorType = "models-data-extractor"
)

var ModelsAttributeKey = plugin.NewDataKey("/v1/models", ModelsExtractorType)

// ModelDataCollection contains model data reported by model-server endpoints.
type ModelDataCollection []ModelData

// ModelData contains one model entry reported by a model server's /v1/models
// API and its internal parent model identifier.
type ModelData struct {
	ID           string `json:"id"`
	Object       string `json:"object,omitempty"`
	Created      int64  `json:"created,omitempty"`
	OwnedBy      string `json:"owned_by,omitempty"`
	ShutdownDate string `json:"shutdown_date,omitempty"`
	Parent       string `json:"parent,omitempty"`
}

// UnmarshalJSON ignores invalid shutdown dates so one malformed field does not reject the model list.
func (m *ModelData) UnmarshalJSON(data []byte) error {
	*m = ModelData{}
	type plain ModelData
	decoded := struct {
		*plain
		ShutdownDate json.RawMessage `json:"shutdown_date"`
	}{plain: (*plain)(m)}
	if err := json.Unmarshal(data, &decoded); err != nil {
		return err
	}

	if len(decoded.ShutdownDate) == 0 {
		return nil
	}
	var shutdownDate string
	if err := json.Unmarshal(decoded.ShutdownDate, &shutdownDate); err != nil {
		return nil
	}
	if _, err := time.Parse(time.DateOnly, shutdownDate); err != nil {
		return nil
	}
	m.ShutdownDate = shutdownDate
	return nil
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
