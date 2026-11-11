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

package requestcontrol

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

func TestParsePriorityClass(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name    string
		headers map[string]string
		want    parsedPriorityClass
	}{
		{
			name: "missing header is critical",
			want: parsedPriorityClass{class: metadata.PriorityClassCritical},
		},
		{
			name:    "critical",
			headers: map[string]string{metadata.PriorityClassKey: "critical"},
			want:    parsedPriorityClass{class: metadata.PriorityClassCritical},
		},
		{
			name:    "critical is case insensitive and trimmed",
			headers: map[string]string{metadata.PriorityClassKey: "  Critical  "},
			want:    parsedPriorityClass{class: metadata.PriorityClassCritical},
		},
		{
			name:    "sheddable",
			headers: map[string]string{metadata.PriorityClassKey: "sheddable"},
			want:    parsedPriorityClass{class: metadata.PriorityClassSheddable},
		},
		{
			name:    "sheddable is case insensitive",
			headers: map[string]string{metadata.PriorityClassKey: "SHEDDABLE"},
			want:    parsedPriorityClass{class: metadata.PriorityClassSheddable},
		},
		{
			name:    "empty value is critical and invalid",
			headers: map[string]string{metadata.PriorityClassKey: ""},
			want:    parsedPriorityClass{class: metadata.PriorityClassCritical, invalid: true},
		},
		{
			name:    "whitespace is critical and invalid",
			headers: map[string]string{metadata.PriorityClassKey: "   "},
			want:    parsedPriorityClass{class: metadata.PriorityClassCritical, invalid: true},
		},
		{
			name:    "unrecognized value is critical and invalid",
			headers: map[string]string{metadata.PriorityClassKey: "excess"},
			want:    parsedPriorityClass{class: metadata.PriorityClassCritical, invalid: true},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, test.want, parsePriorityClass(test.headers))
		})
	}
}

func TestAdmissionPriority(t *testing.T) {
	t.Parallel()

	sheddable := parsedPriorityClass{class: metadata.PriorityClassSheddable}
	critical := parsedPriorityClass{class: metadata.PriorityClassCritical}
	invalid := parsedPriorityClass{class: metadata.PriorityClassCritical, invalid: true}

	tests := []struct {
		name              string
		objectivePriority int
		class             parsedPriorityClass
		want              int
	}{
		{name: "sheddable replaces a positive objective", objectivePriority: 2, class: sheddable, want: sheddableAdmissionPriority},
		{name: "sheddable replaces an already negative objective", objectivePriority: -5, class: sheddable, want: sheddableAdmissionPriority},
		{name: "critical keeps a positive objective", objectivePriority: 2, class: critical, want: 2},
		{name: "critical does not promote a negative objective", objectivePriority: -1, class: critical, want: -1},
		{name: "invalid keeps the objective priority", objectivePriority: 2, class: invalid, want: 2},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()
			assert.Equal(t, test.want, admissionPriority(test.objectivePriority, test.class))
		})
	}
}
