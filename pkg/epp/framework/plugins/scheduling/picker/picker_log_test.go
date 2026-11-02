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

package picker_test

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/go-logr/logr"
	k8stypes "k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/maxscore"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/random"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/weightedrandom"
)

func TestPickLogsScoredEndpointNames(t *testing.T) {
	tests := []struct {
		name   string
		picker fwksched.Picker
	}{
		{name: "max score picker", picker: maxscore.NewMaxScorePicker(1)},
		{name: "random picker", picker: random.NewRandomPicker(1)},
		{name: "weighted random picker", picker: weightedrandom.NewWeightedRandomPicker(1)},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var buf bytes.Buffer
			ctx := log.IntoContext(context.Background(), logutil.NewTestLoggerWithWriter(&buf))
			scoped, _ := datalayer.Scope(logr.Discard(), "Picker", test.picker, []fwksched.Endpoint{
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "ns", Name: "pod1"}}, nil, nil),
				fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Namespace: "ns", Name: "pod2"}}, nil, nil),
			})
			scoredEndpoints := []*fwksched.ScoredEndpoint{{Endpoint: scoped[0], Score: 2}, {Endpoint: scoped[1], Score: 0.5}}

			test.picker.Pick(ctx, scoredEndpoints)

			// Map keys are rendered sorted, so the whole map is deterministic.
			want := `{"ns/pod1":2,"ns/pod2":0.5}`
			if !strings.Contains(buf.String(), want) {
				t.Errorf("log output does not contain %s:\n%s", want, buf.String())
			}
		})
	}
}
