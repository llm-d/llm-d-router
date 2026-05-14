package bylabel

import (
	"context"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-inference-scheduler/pkg/epp/framework/interface/scheduling"
)

const (
	// ByLabelType is the type of the ByLabel filter.
	//
	// Deprecated: Use LabelSelectorFilterType for generic label-based filtering,
	// or the role-specific filters (decode-filter, prefill-filter, encode-filter) for role-based filtering.
	ByLabelType = "by-label"
)

type byLabelParameters struct {
	Label         string   `json:"label"`
	ValidValues   []string `json:"validValues"`
	AllowsNoLabel bool     `json:"allowsNoLabel"`
}

var _ scheduling.Filter = &ByLabel{} // validate interface conformance

// Factory defines the factory function for the ByLabel filter.
//
// Deprecated: Use SelectorFactory for generic label-based filtering,
// or the role-specific filters (decode-filter, prefill-filter, encode-filter) for role-based filtering.
func Factory(name string, rawParameters *json.Decoder, handle plugin.Handle) (plugin.Plugin, error) {
	if handle != nil {
		log.FromContext(handle.Context()).Info("Deprecated: plugin type 'by-label' is deprecated, " +
			"use 'label-selector-filter' for generic label filtering or " +
			"'decode-filter'/'prefill-filter'/'encode-filter' for role-based filtering")
	}
	parameters := byLabelParameters{}
	if rawParameters != nil {
		if err := rawParameters.Decode(&parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' filter - %w", ByLabelType, err)
		}
	}
	if name == "" {
		return nil, fmt.Errorf("invalid configuration for '%s' filter: name cannot be empty", ByLabelType)
	}
	if parameters.Label == "" {
		return nil, fmt.Errorf("invalid configuration for '%s' filter: 'label' must be specified", ByLabelType)
	}
	if len(parameters.ValidValues) == 0 && !parameters.AllowsNoLabel {
		return nil, fmt.Errorf("invalid configuration for '%s' "+
			"filter: either 'validValues' must be non-empty or 'allowsNoLabel' must be true", ByLabelType)
	}
	return NewByLabel(name, parameters.Label, parameters.AllowsNoLabel, parameters.ValidValues...), nil
}

// NewByLabel creates and returns an instance of the ByLabel filter based on the input parameters
// name - the filter name
// labelName - the name of the label to use
// allowsNoLabel - if true endpoints without given label will be considered as valid (not filtered out)
// validValuesApp - list of valid values
func NewByLabel(name string, labelName string, allowsNoLabel bool, validValues ...string) *ByLabel {
	validValuesMap := map[string]struct{}{}

	for _, v := range validValues {
		validValuesMap[v] = struct{}{}
	}

	return &ByLabel{
		typedName:     plugin.TypedName{Type: ByLabelType, Name: name},
		labelName:     labelName,
		allowsNoLabel: allowsNoLabel,
		validValues:   validValuesMap,
	}
}

// ByLabel - filters out endpoints based on the values defined by the given label
type ByLabel struct {
	// name defines the filter typed name
	typedName plugin.TypedName
	// labelName defines the name of the label to be checked
	labelName string
	// validValues defines list of valid label values
	validValues map[string]struct{}
	// allowsNoLabel - if true endpoints without given label will be considered as valid (not filtered out)
	allowsNoLabel bool
}

// TypedName returns the typed name of the plugin
func (f *ByLabel) TypedName() plugin.TypedName {
	return f.typedName
}

// WithName sets the name of the plugin.
func (f *ByLabel) WithName(name string) *ByLabel {
	f.typedName.Name = name
	return f
}

// Filter filters out all endpoints that are not marked with one of roles from the validRoles collection
// or has no role label in case allowsNoRolesLabel is true
func (f *ByLabel) Filter(_ context.Context, _ *scheduling.CycleState, _ *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	filteredEndpoints := []scheduling.Endpoint{}

	for _, endpoint := range endpoints {
		val, labelDefined := endpoint.GetMetadata().Labels[f.labelName]
		_, valueExists := f.validValues[val]

		if (!labelDefined && f.allowsNoLabel) || valueExists {
			filteredEndpoints = append(filteredEndpoints, endpoint)
		}
	}

	return filteredEndpoints
}
