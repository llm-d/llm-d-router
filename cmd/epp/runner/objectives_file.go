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
	"fmt"
	"io"
	"os"
	"path/filepath"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/yaml"

	apixv1 "github.com/llm-d/llm-d-router/apix/v1"
	"github.com/llm-d/llm-d-router/pkg/epp/flowcontrol/contracts"
)

const maxObjectivesFileSize = 1 << 20 // 1 MiB

// objectivesFile is the --objectives-file schema: only the InferenceObjective
// fields a file-discovery EPP uses.
type objectivesFile struct {
	Objectives []objectiveEntry `json:"objectives"`
}

type objectiveEntry struct {
	Name string `json:"name"`
	// Priority defaults to 0, as for an InferenceObjective with no priority.
	Priority int32 `json:"priority,omitempty"`
}

type objectiveSetter interface {
	ObjectiveSet(*apixv1.InferenceObjective)
}

// loadObjectivesFile stores the objectives in path in ds as InferenceObjectives
// and returns their priorities. File-discovery mode has no reconciler to do this.
// Every objective applies to this EPP; there is no pool targeting. An empty path
// is a no-op.
func loadObjectivesFile(path string, ds objectiveSetter) (map[int]struct{}, error) {
	priorities := map[int]struct{}{}
	if path == "" {
		return priorities, nil
	}
	fh, err := os.Open(filepath.Clean(path))
	if err != nil {
		return nil, err
	}
	defer fh.Close()
	data, err := io.ReadAll(io.LimitReader(fh, maxObjectivesFileSize+1))
	if err != nil {
		return nil, fmt.Errorf("reading objectives file %s: %w", path, err)
	}
	if len(data) > maxObjectivesFileSize {
		return nil, fmt.Errorf("objectives file %s exceeds 1 MiB limit", path)
	}
	var f objectivesFile
	if err := yaml.UnmarshalStrict(data, &f); err != nil {
		return nil, fmt.Errorf("objectives file %s: %w", path, err)
	}
	seen := map[string]bool{}
	for i, e := range f.Objectives {
		if e.Name == "" {
			return nil, fmt.Errorf("objectives file %s, objective %d: name is required", path, i)
		}
		if seen[e.Name] {
			return nil, fmt.Errorf("objectives file %s, objective %d: duplicate name %q", path, i, e.Name)
		}
		seen[e.Name] = true
		ds.ObjectiveSet(&apixv1.InferenceObjective{
			ObjectMeta: metav1.ObjectMeta{Name: e.Name},
			Spec:       apixv1.InferenceObjectiveSpec{Priority: ptr.To(e.Priority)},
		})
		priorities[int(e.Priority)] = struct{}{}
	}
	return priorities, nil
}

// provisionObjectiveBands provisions a flow control band for each objective
// priority before the server starts serving. SubmitDesiredPriorities is applied
// asynchronously by the processor loop, so a request arriving in that window
// would find no band and fall back to priority 0. plane is nil when the
// flowControl feature gate is off.
func provisionObjectiveBands(plane contracts.PriorityBandControlPlane, priorities map[int]struct{}) {
	if plane == nil || len(priorities) == 0 {
		return
	}
	if registry, ok := plane.(contracts.FlowRegistryBackground); ok {
		registry.ApplyDesiredPriorities(priorities)
		return
	}
	plane.SubmitDesiredPriorities(priorities)
}
