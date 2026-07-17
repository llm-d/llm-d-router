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

package registry

import (
	"errors"
	"fmt"
	"time"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/flowcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/fairness/globalstrict"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/ordering/fcfs"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/flowcontrol/usagelimits"
)

// --- Defaults ---

const (
	// DefaultOrderingPolicyRef is the default policy for selecting items within a single flow's queue.
	DefaultOrderingPolicyRef string = fcfs.FCFSOrderingPolicyType
	// DefaultFairnessPolicyRef is the default policy for selecting which flow's queue to service next.
	DefaultFairnessPolicyRef string = globalstrict.GlobalStrictFairnessPolicyType
	// DefaultUsageLimitPolicyRef is the default policy to compute usage limit of a priority band dynamically.
	DefaultUsageLimitPolicyRef string = usagelimits.StaticUsageLimitPolicyType
)

const (
	// defaultPriorityBandMaxBytes is the default global capacity for a priority band if not explicitly configured.
	// It is set to 1 GB.
	defaultPriorityBandMaxBytes uint64 = 1_000_000_000
	// defaultFlowGCTimeout is the default duration of inactivity after which an idle flow is garbage collected.
	// This also serves as the interval for the periodic garbage collection scan.
	defaultFlowGCTimeout time.Duration = 5 * time.Minute
	// defaultPriorityBandGCTimeout is the default duration of inactivity after which a dynamically provisioned
	// priority band is garbage collected. Set to 2x flow GC timeout to ensure flows are cleaned up first.
	defaultPriorityBandGCTimeout time.Duration = 2 * defaultFlowGCTimeout
)

// --- Configuration ---

// PriorityBandPolicyDefaults carries pre-resolved default policy instances.
// It is populated by the config loader (the single boundary for plugin resolution)
// and passed into registry constructors so they never need to access the plugin Handle.
type PriorityBandPolicyDefaults struct {
	OrderingPolicy flowcontrol.OrderingPolicy
	FairnessPolicy flowcontrol.FairnessPolicy
}

// Config holds the master configuration for the entire FlowRegistry.
// It serves as the top-level blueprint, defining global settings and the templates for its priority bands.
type Config struct {
	// MaxBytes defines an optional, global maximum total byte size limit aggregated across all priority bands.
	// The `controller.FlowController` enforces this limit in addition to per-band capacity limits.
	// A value of 0 signifies that this global limit is ignored, and only per-band limits apply.
	// Optional: Defaults to 0.
	MaxBytes uint64

	// MaxRequests defines an optional, global maximum total request count aggregated across all priority bands.
	// The `controller.FlowController` enforces this limit in addition to per-band capacity limits.
	// A value of 0 signifies that this global limit is ignored, and only per-band limits apply.
	// Optional: Defaults to 0.
	MaxRequests uint64

	// PriorityBands defines the set of priority band templates managed by the `FlowRegistry`.
	// It is a map keyed by Priority level, providing O(1) access and ensuring priority uniqueness by definition.
	PriorityBands map[int]*PriorityBandConfig

	// DefaultPriorityBand serves as a template for dynamically provisioning priority bands when a request arrives with a
	// priority level that was not explicitly configured.
	// If nil, it is automatically populated with system defaults during NewConfig.
	DefaultPriorityBand *PriorityBandConfig

	// DefaultNegativePriorityBand is an optional template for dynamically provisioning priority bands when a request
	// arrives with a priority level strictly below zero. This allows setting lower capacity limits (or zero) for
	// negative-priority traffic to designate it as sheddable.
	// If nil, negative priorities fall back to DefaultPriorityBand.
	DefaultNegativePriorityBand *PriorityBandConfig

	// FlowGCTimeout defines the interval at which the registry scans for and garbage collects idle flows.
	// A flow is collected if it has been observed to be Idle for at least one full scan interval.
	// Optional: Defaults to `defaultFlowGCTimeout` (1 hour).
	FlowGCTimeout time.Duration

	// PriorityBandGCTimeout defines the duration of inactivity after which a dynamically provisioned priority band
	// is garbage collected. A band is considered idle when it has no flows and no buffered requests.
	// Must be >= FlowGCTimeout to ensure flows are collected before bands.
	// Optional: Defaults to `defaultPriorityBandGCTimeout` (10 minutes).
	PriorityBandGCTimeout time.Duration
}

func (c *Config) String() string {
	if c == nil {
		return "<nil>"
	}
	// Define a local type definition to prevent infinite recursion when calling Sprintf("%+v").
	// A new type definition inherits the struct fields but does not copy its methods,
	// bypassing the Stringer check and allowing a safe reflection-based field dump.
	type temp Config
	return fmt.Sprintf("%+v", temp(*c))
}

// PriorityBandConfig defines the configuration template for a single priority band.
// A "Band" is defined as the collection (or range) of all flows having the same priority level.
// It establishes the default behaviors (such as queueing and dispatch behaviors) and total capacity limits for all flows
// that operate at this priority level.
type PriorityBandConfig struct {
	// Priority is the unique numerical priority level for this band.
	// Convention: Highest numeric value corresponds to highest priority (centered on 0).
	// Required.
	Priority int

	// OrderingPolicy is the hydrated singleton instance of the policy.
	// This policy governs which request *within this flow's queue* to select next (e.g., "fcfs").
	// This field is populated either via WithOrderingPolicy (using a handle lookup) or via applyDefaults.
	// Optional: Defaults to defaultOrderingPolicyRef ("fcfs-ordering-policy").
	OrderingPolicy flowcontrol.OrderingPolicy

	// FairnessPolicy is the hydrated singleton instance of the policy.
	// This policy governs which Flow *within this band* to select next (e.g., "round-robin").
	// This field is populated either via WithFairnessPolicy (using a handle lookup) or via applyDefaults.
	// Optional: Defaults to defaultFairnessPolicyRef ("global-strict-fairness-policy").
	FairnessPolicy flowcontrol.FairnessPolicy

	// MaxBytes defines the maximum total byte size for this priority band.
	// Optional: Defaults to defaultPriorityBandMaxBytes (1 GB).
	MaxBytes uint64

	// MaxRequests defines the maximum total request count for this priority band.
	// A value of 0 signifies no request-count limit is enforced.
	// Optional: Defaults to defaultPriorityBandMaxRequests (5000).
	MaxRequests uint64
}

func (p *PriorityBandConfig) String() string {
	if p == nil {
		return "<nil>"
	}
	// Define a local type definition to prevent infinite recursion when calling Sprintf("%+v").
	// A new type definition inherits the struct fields but does not copy its methods,
	// bypassing the Stringer check and allowing a safe reflection-based field dump.
	type temp PriorityBandConfig
	return fmt.Sprintf("%+v", temp(*p))
}

// --- Config Functional Options ---

// configBuilder holds the intermediate state during NewConfig.
type configBuilder struct {
	config *Config
}

// ConfigOption defines a functional option for configuring the registry.
type ConfigOption func(*configBuilder) error

// WithMaxBytes sets the global maximum total byte size limit.
func WithMaxBytes(maxBytes uint64) ConfigOption {
	return func(b *configBuilder) error {
		b.config.MaxBytes = maxBytes
		return nil
	}
}

// WithMaxRequests sets the global maximum total request count limit.
func WithMaxRequests(maxRequests uint64) ConfigOption {
	return func(b *configBuilder) error {
		b.config.MaxRequests = maxRequests
		return nil
	}
}

// WithFlowGCTimeout sets the idle flow garbage collection interval.
func WithFlowGCTimeout(d time.Duration) ConfigOption {
	return func(b *configBuilder) error {
		if d <= 0 {
			return errors.New("flowGCTimeout must be positive")
		}
		b.config.FlowGCTimeout = d
		return nil
	}
}

// WithPriorityBandGCTimeout sets the idle priority band garbage collection timeout.
func WithPriorityBandGCTimeout(d time.Duration) ConfigOption {
	return func(b *configBuilder) error {
		if d <= 0 {
			return errors.New("priorityBandGCTimeout must be positive")
		}
		if b.config.FlowGCTimeout > 0 && d < b.config.FlowGCTimeout {
			return errors.New("priorityBandGCTimeout must be >= flowGCTimeout")
		}
		b.config.PriorityBandGCTimeout = d
		return nil
	}
}

// WithPriorityBand adds a priority band configuration.
// If a band with the same Priority already exists, it returns an error.
func WithPriorityBand(band *PriorityBandConfig) ConfigOption {
	return func(b *configBuilder) error {
		if band == nil {
			return errors.New("cannot add nil PriorityBandConfig")
		}
		if _, exists := b.config.PriorityBands[band.Priority]; exists {
			return fmt.Errorf("duplicate priority level %d", band.Priority)
		}
		b.config.PriorityBands[band.Priority] = band
		return nil
	}
}

// WithDefaultPriorityBand sets the template configuration used for dynamically provisioning priority bands.
func WithDefaultPriorityBand(band *PriorityBandConfig) ConfigOption {
	return func(b *configBuilder) error {
		b.config.DefaultPriorityBand = band
		return nil
	}
}

// WithDefaultNegativePriorityBand sets the template configuration used for dynamically provisioning priority bands
// with priority levels strictly below zero.
func WithDefaultNegativePriorityBand(band *PriorityBandConfig) ConfigOption {
	return func(b *configBuilder) error {
		b.config.DefaultNegativePriorityBand = band
		return nil
	}
}

// --- PriorityBandConfig Functional Options ---

// PriorityBandConfigOption defines a functional option for configuring a single PriorityBandConfig.
type PriorityBandConfigOption func(*PriorityBandConfig) error

// WithOrderingPolicy sets the ordering policy instance for this priority band.
func WithOrderingPolicy(policy flowcontrol.OrderingPolicy) PriorityBandConfigOption {
	return func(p *PriorityBandConfig) error {
		if policy == nil {
			return errors.New("ordering policy cannot be nil")
		}
		p.OrderingPolicy = policy
		return nil
	}
}

// WithFairnessPolicy sets the fairness policy instance for this priority band.
func WithFairnessPolicy(policy flowcontrol.FairnessPolicy) PriorityBandConfigOption {
	return func(p *PriorityBandConfig) error {
		if policy == nil {
			return errors.New("fairness policy cannot be nil")
		}
		p.FairnessPolicy = policy
		return nil
	}
}

// WithBandMaxBytes sets the capacity limit for this specific priority band.
func WithBandMaxBytes(maxBytes uint64) PriorityBandConfigOption {
	return func(p *PriorityBandConfig) error {
		p.MaxBytes = maxBytes
		return nil
	}
}

// WithBandMaxRequests sets the request count limit for this specific priority band.
func WithBandMaxRequests(maxRequests uint64) PriorityBandConfigOption {
	return func(p *PriorityBandConfig) error {
		p.MaxRequests = maxRequests
		return nil
	}
}

// NewConfig creates a new Config populated with system defaults, applies the provided options, and enforces strict
// validation.
//
// Arguments:
//   - defaults: Default policy instances, provided by the config loader.
//   - opts: Optional configuration overrides.
func NewConfig(defaults PriorityBandPolicyDefaults, opts ...ConfigOption) (*Config, error) {
	builder := &configBuilder{
		config: &Config{
			MaxBytes:              0, // no limit enforced
			MaxRequests:           0, // no limit enforced
			FlowGCTimeout:         defaultFlowGCTimeout,
			PriorityBandGCTimeout: defaultPriorityBandGCTimeout,
			PriorityBands:         make(map[int]*PriorityBandConfig),
		},
	}

	for _, opt := range opts {
		if err := opt(builder); err != nil {
			return nil, err
		}
	}

	// Initialize DefaultPriorityBand if missing.
	// This ensures we always have a template for dynamic provisioning.
	if builder.config.DefaultPriorityBand == nil {
		template, err := NewPriorityBandConfig(0, defaults)
		if err != nil {
			return nil, fmt.Errorf("failed to create default priority band: %w", err)
		}
		builder.config.DefaultPriorityBand = template
	} else {
		if err := builder.config.DefaultPriorityBand.applyDefaults(defaults); err != nil {
			return nil, fmt.Errorf("failed to apply defaults to DefaultPriorityBand: %w", err)
		}
	}

	// Apply defaults to DefaultNegativePriorityBand if set.
	if builder.config.DefaultNegativePriorityBand != nil {
		if err := builder.config.DefaultNegativePriorityBand.applyDefaults(defaults); err != nil {
			return nil, fmt.Errorf("failed to apply defaults to DefaultNegativePriorityBand: %w", err)
		}
	}

	// Apply defaults to all explicitly configured bands.
	for _, band := range builder.config.PriorityBands {
		if err := band.applyDefaults(defaults); err != nil {
			return nil, fmt.Errorf("failed to apply defaults to priority band %d: %w", band.Priority, err)
		}
	}

	// Ensure priority 0 is always provisioned. It is the fallback band for unmatched requests
	// and the demotion target in withConnectionWithDemotion.
	if _, exists := builder.config.PriorityBands[0]; !exists {
		zero := *builder.config.DefaultPriorityBand
		zero.Priority = 0
		if err := zero.applyDefaults(defaults); err != nil {
			return nil, fmt.Errorf("failed to apply defaults to priority band 0: %w", err)
		}
		builder.config.PriorityBands[0] = &zero
	}

	if err := builder.config.validate(); err != nil {
		return nil, fmt.Errorf("invalid registry config: %w", err)
	}
	return builder.config, nil
}

// NewPriorityBandConfig creates a new band configuration with the required fields.
// It applies system defaults first, then applies any provided options to override those defaults.
func NewPriorityBandConfig(
	priority int,
	defaults PriorityBandPolicyDefaults,
	opts ...PriorityBandConfigOption,
) (*PriorityBandConfig, error) {
	pb := &PriorityBandConfig{
		Priority: priority,
	}

	for _, opt := range opts {
		if err := opt(pb); err != nil {
			return nil, err
		}
	}

	if err := pb.applyDefaults(defaults); err != nil {
		return nil, err
	}

	return pb, nil
}

// --- Validation, Defaults & Hydration ---

func (p *PriorityBandConfig) applyDefaults(defaults PriorityBandPolicyDefaults) error {
	if p.OrderingPolicy == nil {
		if defaults.OrderingPolicy == nil {
			return fmt.Errorf("no default ordering policy provided and none set on band %d", p.Priority)
		}
		p.OrderingPolicy = defaults.OrderingPolicy
	}
	if p.MaxBytes == 0 {
		p.MaxBytes = defaultPriorityBandMaxBytes
	}
	if p.FairnessPolicy == nil {
		if defaults.FairnessPolicy == nil {
			return fmt.Errorf("no default fairness policy provided and none set on band %d", p.Priority)
		}
		p.FairnessPolicy = defaults.FairnessPolicy
	}
	return nil
}

// validate checks the integrity of a single band's configuration.
func (p *PriorityBandConfig) validate() error {
	if p.OrderingPolicy == nil {
		return fmt.Errorf("OrderingPolicy instance is missing for priority band %d", p.Priority)
	}
	if p.FairnessPolicy == nil {
		return fmt.Errorf("FairnessPolicy instance is missing for priority band %d", p.Priority)
	}
	return nil
}

// validate checks global constraints and delegates band validation.
func (c *Config) validate() error {
	if c.FlowGCTimeout <= 0 {
		return errors.New("flowGCTimeout must be positive")
	}
	if c.PriorityBandGCTimeout <= 0 {
		return errors.New("priorityBandGCTimeout must be positive")
	}
	if c.PriorityBandGCTimeout < c.FlowGCTimeout {
		return errors.New("priorityBandGCTimeout must be >= flowGCTimeout")
	}

	// Validate the dynamic template.
	// We use a dummy priority since the template itself doesn't have a fixed priority.
	templateValidationCopy := *c.DefaultPriorityBand
	templateValidationCopy.Priority = 0
	if err := templateValidationCopy.validate(); err != nil {
		return fmt.Errorf("invalid DefaultPriorityBand configuration: %w", err)
	}

	if c.DefaultNegativePriorityBand != nil {
		negTemplateCopy := *c.DefaultNegativePriorityBand
		negTemplateCopy.Priority = -1
		if err := negTemplateCopy.validate(); err != nil {
			return fmt.Errorf("invalid DefaultNegativePriorityBand configuration: %w", err)
		}
	}

	// Validate statically configured bands.
	for _, band := range c.PriorityBands {
		if err := band.validate(); err != nil {
			return err
		}
	}
	return nil
}

// Clone creates a deep copy of the Config.
// It ensures the new Config has its own independent map and PriorityBandConfig instances.
func (c *Config) Clone() *Config {
	if c == nil {
		return nil
	}

	clone := *c

	if c.DefaultPriorityBand != nil {
		val := *c.DefaultPriorityBand
		clone.DefaultPriorityBand = &val
	}

	if c.DefaultNegativePriorityBand != nil {
		val := *c.DefaultNegativePriorityBand
		clone.DefaultNegativePriorityBand = &val
	}

	if c.PriorityBands != nil {
		clone.PriorityBands = make(map[int]*PriorityBandConfig, len(c.PriorityBands))
		for prio, band := range c.PriorityBands {
			// Dereference the pointer to copy the struct value, then take the address of the new value.
			// This ensures 'clone' points to a new memory address.
			b := *band
			clone.PriorityBands[prio] = &b
		}
	}
	return &clone
}
