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

package runner

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"

	apixv1 "github.com/llm-d/llm-d-router/apix/v1"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
	fcregistry "github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/registry"
	"github.com/llm-d/llm-d-router/pkg/epp/requestcontrol"
	runserver "github.com/llm-d/llm-d-router/pkg/epp/server"
)

type objectiveStore map[string]int32

func (s objectiveStore) ObjectiveSet(o *apixv1.InferenceObjective) { s[o.Name] = *o.Spec.Priority }

func writeObjectivesFile(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "objectives.yaml")
	require.NoError(t, os.WriteFile(path, []byte(content), 0o644))
	return path
}

func TestLoadObjectivesFile(t *testing.T) {
	path := writeObjectivesFile(t, `objectives:
  - name: interactive
    priority: 10
  - name: batch
    priority: -1
  - name: unset
`)
	store := objectiveStore{}
	priorities, err := loadObjectivesFile(path, store)
	require.NoError(t, err)
	require.Equal(t, objectiveStore{"interactive": 10, "batch": -1, "unset": 0}, store)
	require.Equal(t, map[int]struct{}{10: {}, -1: {}, 0: {}}, priorities)

	priorities, err = loadObjectivesFile("", store)
	require.NoError(t, err)
	require.Empty(t, priorities)
}

func TestLoadObjectivesFileErrors(t *testing.T) {
	for name, tc := range map[string]struct{ content, err string }{
		"missing name":     {"objectives:\n  - priority: 1\n", "objective 0: name is required"},
		"duplicate name":   {"objectives:\n  - name: o\n  - name: o\n    priority: 1\n", `objective 1: duplicate name "o"`},
		"unknown field":    {"objectives:\n  - name: o\n    weight: 1\n", "unknown field"},
		"invalid priority": {"objectives:\n  - name: o\n    priority: high\n", "cannot unmarshal string"},
		"manifest":         {"apiVersion: llm-d.ai/v1\nkind: InferenceObjective\nmetadata:\n  name: o\n", "unknown field"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := loadObjectivesFile(writeObjectivesFile(t, tc.content), objectiveStore{})
			require.ErrorContains(t, err, tc.err)
		})
	}
	_, err := loadObjectivesFile(filepath.Join(t.TempDir(), "missing.yaml"), objectiveStore{})
	require.ErrorIs(t, err, os.ErrNotExist)
}

// TestObjectivesFileWiring follows the file-discovery startup path: objectives are
// loaded into the datastore, and with the flowControl gate their bands exist as
// soon as provisionObjectiveBands returns, before any request can arrive.
func TestObjectivesFileWiring(t *testing.T) {
	path := writeObjectivesFile(t, "objectives:\n  - name: interactive\n    priority: 10\n  - name: batch\n    priority: -1\n")
	for _, gate := range []string{"flowControl", "flowControl=false"} {
		t.Run(gate, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			opts := runserver.NewOptions()
			opts.ConfigText = fmt.Sprintf("apiVersion: llm-d.ai/v1\nkind: EndpointPickerConfig\nfeatureGates:\n- %s\n", gate)
			opts.PoolName = testPoolName

			r := NewRunner()
			r.servingCtx, r.cancelServing = context.WithCancel(context.WithoutCancel(ctx))
			t.Cleanup(r.stopServing)
			rawConfig, err := r.parseConfigurationPhaseOne(ctx, opts)
			require.NoError(t, err)

			ds := datastore.NewDatastore(ctx, r.setupMetricsCollection(opts))
			priorities, err := loadObjectivesFile(path, ds)
			require.NoError(t, err)
			require.Equal(t, int32(10), *ds.ObjectiveGet("interactive").Spec.Priority)
			require.Equal(t, int32(-1), *ds.ObjectiveGet("batch").Spec.Priority)

			eppConfig, err := r.parseConfigurationPhaseTwo(ctx, rawConfig, ds, opts.RefreshMetricsInterval)
			require.NoError(t, err)
			_, _, controlPlane, _ := r.initAdmissionControl(r.servingCtx, opts, eppConfig,
				contracts.EndpointCandidates(requestcontrol.NewDatastoreEndpointCandidates(ds)))
			provisionObjectiveBands(controlPlane, priorities)

			if gate == "flowControl=false" {
				require.Nil(t, controlPlane)
				return
			}
			registry, ok := controlPlane.(*fcregistry.FlowRegistry)
			require.True(t, ok)
			require.Subset(t, registry.AllOrderedPriorityLevels(), []int{10, -1})
		})
	}
}
