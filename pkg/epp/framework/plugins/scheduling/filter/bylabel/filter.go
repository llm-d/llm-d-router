package bylabel

import (
	"context"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

// roleFilterType is the stable internal type reported by the role filters
// (decode-filter, prefill-filter, encode-filter). It is not a registered
// plugin type.
const roleFilterType = "by-label"

var _ scheduling.Filter = &ByLabel{} // validate interface conformance

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
		typedName:     plugin.TypedName{Type: roleFilterType, Name: name},
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
func (f *ByLabel) Filter(_ context.Context, _ *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
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
