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
// Package composite implements a saturation detector that combines the signals of other
// saturation detector plugins. The pool saturation it reports is the maximum across its
// children, so flow control gates dispatch on whichever independent load signal (for
// example in-flight concurrency or scraped queue depth) is the most constrained.
//
// The composite deliberately does not implement the scheduling Filter extension point:
// the config loader only auto-injects a gating detector into scheduling profiles when it
// implements Filter, so per-endpoint filtering stays with the child detectors, which are
// listed in the profiles explicitly.
package composite

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/go-logr/logr"
	"sigs.k8s.io/controller-runtime/pkg/log"

	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

const (
	// MaxSaturationDetectorType is the unique identifier for this plugin.
	MaxSaturationDetectorType = "max-saturation-detector"
)

// parameters is the external configuration schema for the composite detector.
type parameters struct {
	// Detectors names the child saturation detector plugins to combine. Each entry must
	// reference a plugin that appears earlier in the plugins list (so it is already
	// instantiated when this factory runs) and that implements
	// flowcontrol.SaturationDetector.
	Detectors []string `json:"detectors"`
}

// MaxSaturationDetectorFactory instantiates the detector plugin using the provided JSON parameters.
func MaxSaturationDetectorFactory(
	name string,
	params *json.Decoder,
	handle fwkplugin.Handle,
) (fwkplugin.Plugin, error) {
	if handle == nil {
		return nil, errors.New("plugin handle is required")
	}
	var cfg parameters
	if params != nil {
		if err := params.Decode(&cfg); err != nil {
			return nil, fmt.Errorf("failed to unmarshal max saturation detector config: %w", err)
		}
	}
	if len(cfg.Detectors) == 0 {
		return nil, errors.New("max saturation detector requires at least one entry in detectors")
	}

	children := make([]flowcontrol.SaturationDetector, 0, len(cfg.Detectors))
	seen := make(map[string]struct{}, len(cfg.Detectors))
	for _, childName := range cfg.Detectors {
		if _, dup := seen[childName]; dup {
			return nil, fmt.Errorf("duplicate entry in detectors: %s", childName)
		}
		seen[childName] = struct{}{}

		p := handle.Plugin(childName)
		if p == nil {
			return nil, fmt.Errorf(
				"detectors plugin not found: %s (child detectors must be declared before %s in the plugins list)",
				childName, MaxSaturationDetectorType)
		}
		child, ok := p.(flowcontrol.SaturationDetector)
		if !ok {
			return nil, fmt.Errorf("plugin %s does not implement flowcontrol.SaturationDetector", childName)
		}
		children = append(children, child)
	}

	return newDetector(name, cfg.Detectors, children, log.FromContext(handle.Context())), nil
}

// The composite deliberately does not implement fwkplugin.ConsumerPlugin. Its children are
// themselves plugins in the configuration, so the framework already validates and orders their
// data dependencies directly; re-declaring the union here would additionally subject those keys
// to the data graph's layer-order check, which places a plugin implementing no scheduling or
// requestcontrol interface before every producer and rejects the configuration.
var _ flowcontrol.SaturationDetector = &detector{}

// detector combines child saturation detectors, reporting the maximum of their signals.
type detector struct {
	typedName fwkplugin.TypedName
	children  []flowcontrol.SaturationDetector
	// childLabels are the metric label values for the per-child saturation gauge,
	// index-aligned with children: the reference name from the configuration.
	childLabels []string
}

// newDetector creates a new instance of the composite max saturation detector.
// childNames must be index-aligned with children.
func newDetector(
	name string,
	childNames []string,
	children []flowcontrol.SaturationDetector,
	logger logr.Logger,
) *detector {
	typedName := fwkplugin.TypedName{
		Type: MaxSaturationDetectorType,
		Name: name,
	}

	logger.WithName(typedName.String()).V(logutil.DEFAULT).Info("Creating new MaxSaturationDetector",
		"detectors", childNames)

	return &detector{
		typedName:   typedName,
		children:    children,
		childLabels: childNames,
	}
}

// TypedName returns the type and name tuple of this plugin instance.
func (d *detector) TypedName() fwkplugin.TypedName {
	return d.typedName
}

// Saturation returns the maximum saturation reported by the child detectors for the given
// candidate endpoints, so the pool gates as saturated when any single signal is exhausted.
// Each child's value is also exported through the per-detector saturation gauge, labeled by
// the stage named in ctx, letting operators tell which signal is driving the combined value.
func (d *detector) Saturation(ctx context.Context, endpoints []datalayer.Endpoint) float64 {
	stage := flowcontrol.SaturationStageFromContext(ctx)
	var maxSat float64
	for i, child := range d.children {
		sat := child.Saturation(ctx, endpoints)
		metrics.RecordFlowControlDetectorSaturation(d.childLabels[i], stage, sat)
		if i == 0 || sat > maxSat {
			maxSat = sat
		}
	}
	return maxSat
}
