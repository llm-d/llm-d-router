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

package metrics

import (
	"errors"
	"fmt"
	"strings"
)

// Mapping holds specifications for the well-known metrics defined
// in the Model Server Protocol.
type Mapping struct {
	TotalQueuedRequests  *Spec
	TotalRunningRequests *Spec
	KVCacheUtilization   *Spec
	LoraRequestInfo      *LoRASpec
	// CacheInfo is used for info-style gauge metrics where block_size and
	// num_gpu_blocks are exposed as label values (e.g. vLLM, trtllm-serve, SGLang).
	CacheInfo *Spec
	// CacheBlockSizeLabel and CacheNumBlocksLabel allow engines to use different
	// label names for the CacheInfo metric. If empty, defaults to "block_size"
	// and "num_gpu_blocks".
	CacheBlockSizeLabel string
	CacheNumBlocksLabel string
	// CacheBlockSize and CacheNumBlocks are used for engines that expose cache
	// config as separate gauge values rather than labels on an info metric
	// (e.g. Triton TRT-LLM).
	CacheBlockSize *Spec
	CacheNumBlocks *Spec
	// TotalCacheTokens lists gauges reporting KV cache capacity in tokens.
	// The extractor takes the maximum across the specs present in a scrape,
	// so tiered engines can list per-tier capacity gauges (e.g. SGLang's
	// hicache host pool, which is an inclusive superset of the device pool).
	// Specs whose metric is absent are skipped rather than treated as errors,
	// since tier gauges only exist when the corresponding feature is enabled.
	TotalCacheTokens []*Spec
	// OffloadDetection lists metrics whose presence indicates that the engine
	// offloads KV cache to another tier without reporting that tier's
	// capacity (e.g. vLLM's OffloadingConnector runtime metrics).
	OffloadDetection []*Spec
	// OffloadSizeLabel optionally names a label on the CacheInfo metric whose
	// numeric value, when greater than zero, likewise indicates KV cache
	// offloading (e.g. vLLM's kv_offloading_size, in GiB).
	OffloadSizeLabel string
	CustomMetrics    []CustomMetric
}

// MappingConfig holds configuration used to build a Mapping.
type MappingConfig struct {
	Queue               string
	Running             string
	KVUsage             string
	Lora                string
	CacheInfo           string
	CacheBlockSizeLabel string
	CacheNumBlocksLabel string
	CacheBlockSize      string
	CacheNumBlocks      string
	TotalCacheTokens    []string
	OffloadDetection    []string
	OffloadSizeLabel    string
	CustomMetrics       []CustomMetric
}

type CustomMetric struct {
	AttributeKey string
	Spec         *Spec
}

type namedSpec struct {
	name    string
	spec    *Spec
	enabled bool
}

func (m *Mapping) specs() []namedSpec {
	var loraSpec *Spec
	if m.LoraRequestInfo != nil {
		loraSpec = m.LoraRequestInfo.Spec
	}
	specs := make([]namedSpec, 0, 5+len(m.TotalCacheTokens)+len(m.OffloadDetection)+len(m.CustomMetrics))
	specs = append(specs,
		namedSpec{"queue", m.TotalQueuedRequests, m.TotalQueuedRequests != nil},
		namedSpec{"running", m.TotalRunningRequests, m.TotalRunningRequests != nil},
		namedSpec{"kv", m.KVCacheUtilization, m.KVCacheUtilization != nil},
		namedSpec{"lora", loraSpec, m.LoraRequestInfo != nil},
		namedSpec{"cacheInfo", m.CacheInfo, m.CacheInfo != nil},
	)
	for _, spec := range m.TotalCacheTokens {
		specs = append(specs, namedSpec{"totalCacheTokens", spec, spec != nil})
	}
	for _, spec := range m.OffloadDetection {
		specs = append(specs, namedSpec{"offloadDetection", spec, spec != nil})
	}
	for _, custom := range m.CustomMetrics {
		specs = append(specs, namedSpec{
			name:    custom.AttributeKey,
			spec:    custom.Spec,
			enabled: custom.Spec != nil,
		})
	}
	return specs
}

// String returns a human-readable representation of the Mapping, listing which specs are disabled (nil).
func (m *Mapping) String() string {
	var disabled []string
	for _, s := range m.specs() {
		if !s.enabled {
			disabled = append(disabled, s.name)
		}
	}
	if len(disabled) == 0 {
		return "Mapping{all specs enabled}"
	}
	return fmt.Sprintf("Mapping{disabled: [%s]}", strings.Join(disabled, ", "))
}

// MetricNames returns the Prometheus metric names for all enabled specs.
func (m *Mapping) MetricNames() []string {
	var names []string
	for _, s := range m.specs() {
		if s.enabled {
			names = append(names, s.spec.Name)
		}
	}
	return names
}

// NewMapping creates a metrics.Mapping from the input specification strings.
func NewMapping(queue, running, kvusage, lora, cacheInfo string) (*Mapping, error) {
	return NewMappingFromConfig(MappingConfig{
		Queue:     queue,
		Running:   running,
		KVUsage:   kvusage,
		Lora:      lora,
		CacheInfo: cacheInfo,
	})
}

// NewMappingFromConfig creates a metrics.Mapping from a MappingConfig.
func NewMappingFromConfig(cfg MappingConfig) (*Mapping, error) {
	var errs []error

	queueSpec, err := parseStringToSpec(cfg.Queue)
	if err != nil {
		errs = append(errs, err)
	}
	runningSpec, err := parseStringToSpec(cfg.Running)
	if err != nil {
		errs = append(errs, err)
	}
	kvusageSpec, err := parseStringToSpec(cfg.KVUsage)
	if err != nil {
		errs = append(errs, err)
	}
	loraSpec, err := parseStringToLoRASpec(cfg.Lora)
	if err != nil {
		errs = append(errs, err)
	}
	cacheInfoSpec, err := parseStringToSpec(cfg.CacheInfo)
	if err != nil {
		errs = append(errs, err)
	}
	cacheBlockSizeSpec, err := parseStringToSpec(cfg.CacheBlockSize)
	if err != nil {
		errs = append(errs, err)
	}
	cacheNumBlocksSpec, err := parseStringToSpec(cfg.CacheNumBlocks)
	if err != nil {
		errs = append(errs, err)
	}
	totalCacheTokensSpecs, specErrs := parseSpecList(cfg.TotalCacheTokens)
	errs = append(errs, specErrs...)
	offloadDetectionSpecs, specErrs := parseSpecList(cfg.OffloadDetection)
	errs = append(errs, specErrs...)
	customMetrics, customErrs := parseCustomMetrics(cfg.CustomMetrics)
	errs = append(errs, customErrs...)

	if len(errs) != 0 {
		return nil, errors.Join(errs...)
	}
	return &Mapping{
		TotalQueuedRequests:  queueSpec,
		TotalRunningRequests: runningSpec,
		KVCacheUtilization:   kvusageSpec,
		LoraRequestInfo:      loraSpec,
		CacheInfo:            cacheInfoSpec,
		CacheBlockSizeLabel:  cfg.CacheBlockSizeLabel,
		CacheNumBlocksLabel:  cfg.CacheNumBlocksLabel,
		CacheBlockSize:       cacheBlockSizeSpec,
		CacheNumBlocks:       cacheNumBlocksSpec,
		TotalCacheTokens:     totalCacheTokensSpecs,
		OffloadDetection:     offloadDetectionSpecs,
		OffloadSizeLabel:     cfg.OffloadSizeLabel,
		CustomMetrics:        customMetrics,
	}, nil
}

// parseSpecList parses a list of metric specification strings, dropping empties.
func parseSpecList(specStrings []string) ([]*Spec, []error) {
	specs := make([]*Spec, 0, len(specStrings))
	var errs []error
	for _, s := range specStrings {
		spec, err := parseStringToSpec(s)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		if spec != nil {
			specs = append(specs, spec)
		}
	}
	return specs, errs
}

func parseCustomMetrics(configs []CustomMetric) ([]CustomMetric, []error) {
	metrics := make([]CustomMetric, 0, len(configs))
	var errs []error
	seenKeys := make(map[string]struct{}, len(configs))
	for _, cfg := range configs {
		if cfg.AttributeKey == "" {
			errs = append(errs, errors.New("custom metric attributeKey cannot be empty"))
			continue
		}
		if _, ok := seenKeys[cfg.AttributeKey]; ok {
			errs = append(errs, fmt.Errorf("custom metric attributeKey %q is duplicated", cfg.AttributeKey))
			continue
		}
		seenKeys[cfg.AttributeKey] = struct{}{}
		if cfg.Spec == nil {
			errs = append(errs, fmt.Errorf("custom metric %q spec cannot be empty", cfg.AttributeKey))
			continue
		}
		metrics = append(metrics, CustomMetric{
			AttributeKey: cfg.AttributeKey,
			Spec:         cfg.Spec,
		})
	}
	return metrics, errs
}
