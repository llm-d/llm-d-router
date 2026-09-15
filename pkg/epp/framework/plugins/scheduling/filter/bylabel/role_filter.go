package bylabel

import (
	"context"
	"encoding/json"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// DecodeRoleType is the plugin type of the filter selecting decode endpoints
	DecodeRoleType = "decode-filter"
	// PrefillRoleType is the plugin type of the filter selecting prefill endpoints
	PrefillRoleType = "prefill-filter"
	// EncodeRoleType is the plugin type of the filter selecting encode endpoints
	EncodeRoleType = "encode-filter"
)

var _ scheduling.Filter = &RoleFilter{} // validate interface conformance

// RoleFilter retains endpoints whose RoleLabel value is one of a fixed set of roles.
type RoleFilter struct {
	typedName plugin.TypedName
	// validRoles defines the set of accepted RoleLabel values
	validRoles map[string]struct{}
	// allowsNoRole - if true endpoints without RoleLabel are retained
	allowsNoRole bool
}

// newRoleFilter returns a filter typed and named after the role it selects for.
func newRoleFilter(roleType string, allowsNoRole bool, validRoles ...string) *RoleFilter {
	validRolesSet := make(map[string]struct{}, len(validRoles))
	for _, role := range validRoles {
		validRolesSet[role] = struct{}{}
	}

	return &RoleFilter{
		typedName:    plugin.TypedName{Type: roleType, Name: roleType},
		validRoles:   validRolesSet,
		allowsNoRole: allowsNoRole,
	}
}

// TypedName returns the typed name of the plugin
func (f *RoleFilter) TypedName() plugin.TypedName {
	return f.typedName
}

// WithName sets the name of the plugin.
func (f *RoleFilter) WithName(name string) *RoleFilter {
	f.typedName.Name = name
	return f
}

// Filter retains the endpoints carrying one of the filter's valid roles, plus those
// carrying no role at all when allowsNoRole is set.
func (f *RoleFilter) Filter(_ context.Context, _ *scheduling.InferenceRequest, endpoints []scheduling.Endpoint) []scheduling.Endpoint {
	filteredEndpoints := []scheduling.Endpoint{}

	for _, endpoint := range endpoints {
		role, roleDefined := endpoint.GetMetadata().Labels[RoleLabel]
		_, roleValid := f.validRoles[role]

		if (!roleDefined && f.allowsNoRole) || roleValid {
			filteredEndpoints = append(filteredEndpoints, endpoint)
		}
	}

	return filteredEndpoints
}

// DecodeRoleFactory defines the factory function for the Decode filter.
func DecodeRoleFactory(name string, _ *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	return NewDecodeRole().WithName(name), nil
}

// NewDecodeRole creates and returns an instance of the Filter configured for decode role.
func NewDecodeRole() *RoleFilter {
	return newRoleFilter(DecodeRoleType, true, RoleDecode, RolePrefillDecode, RoleEncodePrefillDecode)
}

// PrefillRoleFactory defines the factory function for the Prefill filter.
func PrefillRoleFactory(name string, _ *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	return NewPrefillRole().WithName(name), nil
}

// NewPrefillRole creates and returns an instance of the Filter configured for prefill role.
func NewPrefillRole() *RoleFilter {
	return newRoleFilter(PrefillRoleType, false, RolePrefill, RoleEncodePrefill, RolePrefillDecode, RoleEncodePrefillDecode)
}

// EncodeRoleFactory defines the factory function for the Encode filter.
func EncodeRoleFactory(name string, _ *json.Decoder, _ plugin.Handle) (plugin.Plugin, error) {
	return NewEncodeRole().WithName(name), nil
}

// NewEncodeRole creates and returns an instance of the Filter configured for encode role.
func NewEncodeRole() *RoleFilter {
	return newRoleFilter(EncodeRoleType, false, RoleEncode, RoleEncodePrefill, RoleEncodePrefillDecode)
}
