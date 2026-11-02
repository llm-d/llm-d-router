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

package bylabel

import (
	"errors"
	"fmt"
	"maps"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
)

const (
	// RoleTransferGroupLabel identifies workers that may exchange P/D/M roles.
	RoleTransferGroupLabel = "llm-d.ai/role-transfer-group"
	// TensorParallelSizeLabel records the tensor parallel size of a model server Pod.
	TensorParallelSizeLabel = "llm-d.ai/tensor-parallel-size"
	// ModelIdentityLabel records the canonical model and revision identity.
	ModelIdentityLabel = "llm-d.ai/model-identity"
	// AcceleratorClassLabel records the worker hardware compatibility class.
	AcceleratorClassLabel = "llm-d.ai/accelerator-class"
	// EngineProfileRefLabel points to the complete engine configuration descriptor.
	EngineProfileRefLabel = "llm-d.ai/engine-profile"
	// RoleCapabilitiesLabel lists the P/D/M roles supported by the worker.
	RoleCapabilitiesLabel = "llm-d.ai/role-capabilities"
	// MaxModelLenLabel records the endpoint context-length limit.
	MaxModelLenLabel = "llm-d.ai/max-model-len"
	// MaxNumBatchedTokensLabel records the endpoint batch-token limit.
	MaxNumBatchedTokensLabel = "llm-d.ai/max-num-batched-tokens"
	// MaxNumSeqsLabel records the endpoint concurrent-sequence limit.
	MaxNumSeqsLabel = "llm-d.ai/max-num-seqs"
)

var layoutComponentPattern = regexp.MustCompile(`^(\d+)tp(\d+)$`)

type layoutEndpoint struct {
	name    string
	address string
	role    string
	shape   WorkerShape
}

// WorkerShape describes the compatibility and capacity properties of a worker.
type WorkerShape struct {
	TransferGroupID     string   `json:"transfer_group_id"`
	TensorParallelSize  int      `json:"tensor_parallel_size"`
	ModelIdentity       string   `json:"model_identity"`
	AcceleratorClass    string   `json:"accelerator_class"`
	EngineProfileRef    string   `json:"engine_profile_ref,omitempty"`
	RoleCapabilities    []string `json:"role_capabilities,omitempty"`
	MaxModelLen         int      `json:"max_model_len,omitempty"`
	MaxNumBatchedTokens int      `json:"max_num_batched_tokens,omitempty"`
	MaxNumSeqs          int      `json:"max_num_seqs,omitempty"`
}

// LayoutEndpoint describes an endpoint participating in dynamic role assignment.
type LayoutEndpoint struct {
	Name    string `json:"name"`
	Address string `json:"address"`
	WorkerShape
}

func (endpoint layoutEndpoint) toAPI() LayoutEndpoint {
	shape := endpoint.shape
	shape.RoleCapabilities = append([]string(nil), shape.RoleCapabilities...)
	return LayoutEndpoint{Name: endpoint.name, Address: endpoint.address, WorkerShape: shape}
}

// LayoutState is the current role assignment and its compatible layouts.
type LayoutState struct {
	CurrentLayout       string                      `json:"current_layout"`
	GroupIndex          int                         `json:"group_idx"`
	TransferableLayouts []string                    `json:"transferable_layouts"`
	CurrentInstances    map[string][]LayoutEndpoint `json:"current_instances"`
}

// LayoutChange describes an applied or skipped layout update.
type LayoutChange struct {
	Status         string         `json:"status"`
	Layout         string         `json:"layout"`
	PreviousLayout string         `json:"previous_layout"`
	GroupIndex     int            `json:"group_idx"`
	Counts         map[string]int `json:"counts"`
}

// LayoutManager keeps runtime role assignments separate from discovered Pod metadata.
type LayoutManager struct {
	mu        sync.RWMutex
	endpoints map[string]layoutEndpoint
	overrides map[string]string
}

// NewLayoutManager creates an empty dynamic layout manager.
func NewLayoutManager() *LayoutManager {
	return &LayoutManager{
		endpoints: map[string]layoutEndpoint{},
		overrides: map[string]string{},
	}
}

func endpointRole(role string) bool {
	return role == RolePrefill || role == RoleDecode || role == RolePrefillDecode
}

func optionalPositiveInt(labels map[string]string, key string) (int, bool) {
	value := labels[key]
	if value == "" {
		return 0, true
	}
	parsed, err := strconv.Atoi(value)
	return parsed, err == nil && parsed > 0
}

func workerShape(labels map[string]string) (WorkerShape, bool) {
	tp, err := strconv.Atoi(labels[TensorParallelSizeLabel])
	groupID := labels[RoleTransferGroupLabel]
	modelIdentity := labels[ModelIdentityLabel]
	acceleratorClass := labels[AcceleratorClassLabel]
	if err != nil || tp < 1 || groupID == "" || modelIdentity == "" || acceleratorClass == "" {
		return WorkerShape{}, false
	}
	maxModelLen, valid := optionalPositiveInt(labels, MaxModelLenLabel)
	if !valid {
		return WorkerShape{}, false
	}
	maxNumBatchedTokens, valid := optionalPositiveInt(labels, MaxNumBatchedTokensLabel)
	if !valid {
		return WorkerShape{}, false
	}
	maxNumSeqs, valid := optionalPositiveInt(labels, MaxNumSeqsLabel)
	if !valid {
		return WorkerShape{}, false
	}
	capabilities := []string{}
	for _, capability := range strings.Split(labels[RoleCapabilitiesLabel], ",") {
		if capability = strings.TrimSpace(capability); capability != "" {
			capabilities = append(capabilities, capability)
		}
	}
	return WorkerShape{
		TransferGroupID:     groupID,
		TensorParallelSize:  tp,
		ModelIdentity:       modelIdentity,
		AcceleratorClass:    acceleratorClass,
		EngineProfileRef:    labels[EngineProfileRefLabel],
		RoleCapabilities:    capabilities,
		MaxModelLen:         maxModelLen,
		MaxNumBatchedTokens: maxNumBatchedTokens,
		MaxNumSeqs:          maxNumSeqs,
	}, true
}

func discoveredEndpoints(endpoints []scheduling.Endpoint) map[string]layoutEndpoint {
	discovered := make(map[string]layoutEndpoint, len(endpoints))
	for _, endpoint := range endpoints {
		metadata := endpoint.GetMetadata()
		role := metadata.Labels[RoleLabel]
		if !endpointRole(role) {
			continue
		}
		shape, valid := workerShape(metadata.Labels)
		if !valid {
			continue
		}
		name := metadata.ID.String()
		discovered[name] = layoutEndpoint{name: name, address: metadata.Address, role: role, shape: shape}
	}
	return discovered
}

// Observe refreshes the set of endpoints eligible for P/D/M assignment.
func (m *LayoutManager) Observe(endpoints []scheduling.Endpoint) {
	discovered := discoveredEndpoints(endpoints)
	m.mu.Lock()
	m.observeLocked(discovered)
	m.mu.Unlock()
}

func (m *LayoutManager) observeLocked(discovered map[string]layoutEndpoint) {
	m.endpoints = discovered
	for name := range m.overrides {
		if _, exists := discovered[name]; !exists {
			delete(m.overrides, name)
		}
	}
}

// Roles observes endpoints and returns one consistent runtime role snapshot.
func (m *LayoutManager) Roles(endpoints []scheduling.Endpoint) map[string]string {
	discovered := discoveredEndpoints(endpoints)
	m.mu.Lock()
	defer m.mu.Unlock()
	m.observeLocked(discovered)

	roles := make(map[string]string, len(endpoints))
	for _, endpoint := range endpoints {
		metadata := endpoint.GetMetadata()
		name := metadata.ID.String()
		if role, overridden := m.overrides[name]; overridden {
			roles[name] = role
		} else {
			roles[name] = metadata.Labels[RoleLabel]
		}
	}
	return roles
}

func parseLayout(layout string) (map[string][]int, error) {
	parts := strings.Split(layout, "_")
	if len(parts) != 6 || parts[0] != "p" || parts[2] != "d" || parts[4] != "m" {
		return nil, errors.New("layout must use the form p_<components>_d_<components>_m_<components>")
	}

	result := map[string][]int{RolePrefill: {}, RoleDecode: {}, RolePrefillDecode: {}}
	for index, role := range []string{RolePrefill, RoleDecode, RolePrefillDecode} {
		components := parts[index*2+1]
		if components == "none" {
			continue
		}
		for _, component := range strings.Split(components, "-") {
			match := layoutComponentPattern.FindStringSubmatch(component)
			if match == nil {
				return nil, fmt.Errorf("invalid layout component %q", component)
			}
			count, _ := strconv.Atoi(match[1])
			tp, _ := strconv.Atoi(match[2])
			if count < 1 || tp < 1 {
				return nil, errors.New("layout component values must be positive")
			}
			for range count {
				result[role] = append(result[role], tp)
			}
		}
	}
	return result, nil
}

func formatLayout(counts map[string]map[int]int) string {
	parts := make([]string, 0, 6)
	for _, role := range []struct{ short, name string }{{"p", RolePrefill}, {"d", RoleDecode}, {"m", RolePrefillDecode}} {
		parts = append(parts, role.short)
		tps := make([]int, 0, len(counts[role.name]))
		for tp, count := range counts[role.name] {
			if count > 0 {
				tps = append(tps, tp)
			}
		}
		sort.Ints(tps)
		components := make([]string, 0, len(tps))
		for _, tp := range tps {
			components = append(components, fmt.Sprintf("%dtp%d", counts[role.name][tp], tp))
		}
		if len(components) == 0 {
			components = append(components, "none")
		}
		parts = append(parts, strings.Join(components, "-"))
	}
	return strings.Join(parts, "_")
}

func effectiveRole(endpoint layoutEndpoint, overrides map[string]string) string {
	if role, exists := overrides[endpoint.name]; exists {
		return role
	}
	return endpoint.role
}

func currentCounts(endpoints map[string]layoutEndpoint, overrides map[string]string) map[string]map[int]int {
	counts := map[string]map[int]int{RolePrefill: {}, RoleDecode: {}, RolePrefillDecode: {}}
	for _, endpoint := range endpoints {
		counts[effectiveRole(endpoint, overrides)][endpoint.shape.TensorParallelSize]++
	}
	return counts
}

func targetCounts(layout map[string][]int) map[string]map[int]int {
	counts := map[string]map[int]int{RolePrefill: {}, RoleDecode: {}, RolePrefillDecode: {}}
	for role, tps := range layout {
		for _, tp := range tps {
			counts[role][tp]++
		}
	}
	return counts
}

func singleTransferGroup(endpoints map[string]layoutEndpoint) (string, error) {
	groupID := ""
	var reference WorkerShape
	for _, endpoint := range endpoints {
		if groupID == "" {
			groupID = endpoint.shape.TransferGroupID
			reference = endpoint.shape
			continue
		}
		if endpoint.shape.TransferGroupID != groupID {
			return "", errors.New("flat layout API requires exactly one role transfer group")
		}
		if endpoint.shape.ModelIdentity != reference.ModelIdentity ||
			endpoint.shape.AcceleratorClass != reference.AcceleratorClass {
			return "", fmt.Errorf("role transfer group %q has inconsistent model or hardware identity", groupID)
		}
	}
	return groupID, nil
}

func validRoleCombination(layout map[string][]int) bool {
	prefillCount := len(layout[RolePrefill])
	decodeCount := len(layout[RoleDecode])
	mixedCount := len(layout[RolePrefillDecode])
	return (prefillCount > 0 && decodeCount > 0) ||
		(prefillCount == 0 && decodeCount == 0 && mixedCount > 0)
}

func transferableLayouts(endpoints map[string]layoutEndpoint) []string {
	byTP := map[int]int{}
	for _, endpoint := range endpoints {
		byTP[endpoint.shape.TensorParallelSize]++
	}
	tps := make([]int, 0, len(byTP))
	for tp := range byTP {
		tps = append(tps, tp)
	}
	sort.Ints(tps)

	counts := map[string]map[int]int{RolePrefill: {}, RoleDecode: {}, RolePrefillDecode: {}}
	layouts := []string{}
	var generate func(int, int, int, int)
	generate = func(index, totalPrefill, totalDecode, totalMixed int) {
		if index == len(tps) {
			if (totalPrefill > 0 && totalDecode > 0) || (totalPrefill == 0 && totalDecode == 0 && totalMixed > 0) {
				layouts = append(layouts, formatLayout(counts))
			}
			return
		}
		tp := tps[index]
		for prefill := 0; prefill <= byTP[tp]; prefill++ {
			for decode := 0; decode <= byTP[tp]-prefill; decode++ {
				mixed := byTP[tp] - prefill - decode
				counts[RolePrefill][tp] = prefill
				counts[RoleDecode][tp] = decode
				counts[RolePrefillDecode][tp] = mixed
				generate(index+1, totalPrefill+prefill, totalDecode+decode, totalMixed+mixed)
			}
		}
	}
	generate(0, 0, 0, 0)
	sort.Strings(layouts)
	return layouts
}

// Apply atomically reassigns known endpoints to the requested compatible layout.
func (m *LayoutManager) Apply(layout string) (LayoutChange, error) {
	target, err := parseLayout(layout)
	if err != nil {
		return LayoutChange{}, err
	}
	if !validRoleCombination(target) {
		return LayoutChange{}, errors.New("layout must contain both prefill and decode endpoints, or only mixed endpoints")
	}
	// Canonical ordering makes equivalent component strings idempotent, such as
	// d_1tp4-4tp1 and d_4tp1-1tp4.
	layout = formatLayout(targetCounts(target))

	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.endpoints) == 0 {
		return LayoutChange{}, fmt.Errorf("no P/D endpoints with required labels %s, %s, %s, and %s are known", RoleTransferGroupLabel, TensorParallelSizeLabel, ModelIdentityLabel, AcceleratorClassLabel)
	}
	if _, err := singleTransferGroup(m.endpoints); err != nil {
		return LayoutChange{}, err
	}

	previous := formatLayout(currentCounts(m.endpoints, m.overrides))
	availableByTP := map[int]int{}
	requestedByTP := map[int]int{}
	for _, endpoint := range m.endpoints {
		availableByTP[endpoint.shape.TensorParallelSize]++
	}
	for _, role := range []string{RolePrefill, RoleDecode, RolePrefillDecode} {
		for _, tp := range target[role] {
			requestedByTP[tp]++
		}
	}
	// A role may change, but an endpoint's TP topology cannot. Preserve the
	// endpoint count independently in every TP bucket.
	if !maps.Equal(availableByTP, requestedByTP) {
		return LayoutChange{}, errors.New("layout TP allocation does not match discovered endpoints")
	}

	if layout == previous {
		return LayoutChange{Status: "skipped", Layout: layout, PreviousLayout: previous, GroupIndex: 1, Counts: layoutRoleCounts(target)}, nil
	}

	byRoleTP := map[string]map[int][]string{RolePrefill: {}, RoleDecode: {}, RolePrefillDecode: {}}
	for name, endpoint := range m.endpoints {
		role := effectiveRole(endpoint, m.overrides)
		byRoleTP[role][endpoint.shape.TensorParallelSize] = append(byRoleTP[role][endpoint.shape.TensorParallelSize], name)
	}
	for _, byTP := range byRoleTP {
		for tp := range byTP {
			sort.Strings(byTP[tp])
		}
	}

	assignments := map[string]string{}
	released := map[int][]string{}
	remaining := map[string]map[int]int{RolePrefill: {}, RoleDecode: {}, RolePrefillDecode: {}}
	for _, role := range []string{RolePrefill, RoleDecode, RolePrefillDecode} {
		for _, tp := range target[role] {
			remaining[role][tp]++
		}
		for tp, names := range byRoleTP[role] {
			keep := min(len(names), remaining[role][tp])
			for _, name := range names[:keep] {
				assignments[name] = role
			}
			remaining[role][tp] -= keep
			released[tp] = append(released[tp], names[keep:]...)
		}
	}
	for tp := range released {
		sort.Strings(released[tp])
	}
	// Fill each role deficit only from endpoints released by the same TP
	// bucket; for example, a TP4 endpoint can change P -> D but never TP4 -> TP1.
	for _, role := range []string{RolePrefill, RoleDecode, RolePrefillDecode} {
		for tp, count := range remaining[role] {
			if len(released[tp]) < count {
				return LayoutChange{}, fmt.Errorf("not enough tp%d endpoints for role %s", tp, role)
			}
			for _, name := range released[tp][:count] {
				assignments[name] = role
			}
			released[tp] = released[tp][count:]
		}
	}
	m.overrides = assignments
	return LayoutChange{Status: "ok", Layout: layout, PreviousLayout: previous, GroupIndex: 1, Counts: layoutRoleCounts(target)}, nil
}

func layoutRoleCounts(layout map[string][]int) map[string]int {
	return map[string]int{
		"prefill": len(layout[RolePrefill]),
		"decode":  len(layout[RoleDecode]),
		"mixed":   len(layout[RolePrefillDecode]),
	}
}

// State returns the current assignment and all layouts reachable with the known TP buckets.
func (m *LayoutManager) State() (LayoutState, error) {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if len(m.endpoints) == 0 {
		return LayoutState{}, fmt.Errorf("no P/D endpoints with required labels %s, %s, %s, and %s are known", RoleTransferGroupLabel, TensorParallelSizeLabel, ModelIdentityLabel, AcceleratorClassLabel)
	}
	if _, err := singleTransferGroup(m.endpoints); err != nil {
		return LayoutState{}, err
	}
	instances := map[string][]LayoutEndpoint{"prefill": {}, "decode": {}, "mixed": {}}
	roleNames := map[string]string{RolePrefill: "prefill", RoleDecode: "decode", RolePrefillDecode: "mixed"}
	for _, endpoint := range m.endpoints {
		role := roleNames[effectiveRole(endpoint, m.overrides)]
		instances[role] = append(instances[role], endpoint.toAPI())
	}
	for role := range instances {
		sort.Slice(instances[role], func(i, j int) bool { return instances[role][i].Name < instances[role][j].Name })
	}
	return LayoutState{
		CurrentLayout:       formatLayout(currentCounts(m.endpoints, m.overrides)),
		GroupIndex:          1,
		TransferableLayouts: transferableLayouts(m.endpoints),
		CurrentInstances:    instances,
	}, nil
}
