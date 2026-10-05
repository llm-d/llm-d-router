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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestModelDataUnmarshalShutdownDate(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "valid date",
			body: `{"id":"base","shutdown_date":"2026-10-23"}`,
			want: "2026-10-23",
		},
		{
			name: "number is ignored",
			body: `{"id":"base","shutdown_date":1767225600}`,
		},
		{
			name: "invalid date is ignored",
			body: `{"id":"base","shutdown_date":"not-a-date"}`,
		},
		{
			name: "null is ignored",
			body: `{"id":"base","shutdown_date":null}`,
		},
		{
			name: "omitted",
			body: `{"id":"base"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var model ModelData
			require.NoError(t, json.Unmarshal([]byte(tt.body), &model))
			assert.Equal(t, tt.want, model.ShutdownDate)
		})
	}
}

func TestModelDataCollectionCloneCopiesShutdownDate(t *testing.T) {
	original := ModelDataCollection{{ID: "base", ShutdownDate: "2026-10-23"}}
	clone := original.Clone().(ModelDataCollection)

	clone[0].ShutdownDate = "2027-10-23"

	assert.Equal(t, "2026-10-23", original[0].ShutdownDate)
}
