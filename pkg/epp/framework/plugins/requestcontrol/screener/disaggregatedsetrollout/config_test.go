/*
Copyright 2026 The Kubernetes Authors.

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

package disaggregatedsetrollout

import (
	"encoding/json"
	"strings"
	"testing"
)

func validConfig() Config {
	return Config{
		Scope: Scope{
			LabelSelector: "disaggregatedset.x-k8s.io/name=my-set",
		},
		RevisionGating: &RevisionGating{
			RevisionHeaderName: "x-llm-d-disagg-revision",
			RevisionLabelKey:   "disaggregatedset.x-k8s.io/revision",
			RoleLabelKey:       "disaggregatedset.x-k8s.io/role",
			Mode:               GatingModeSum,
			RequiredRoles:      []string{"prefill", "decode"},
		},
	}
}

func TestValidate_Valid(t *testing.T) {
	cfg := validConfig()
	if err := cfg.Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
}

func TestValidate_MissingScopeSelector(t *testing.T) {
	cfg := validConfig()
	cfg.Scope.LabelSelector = ""
	assertValidateError(t, cfg, "scope.labelSelector")
}

func TestValidate_UnparsableScopeSelector(t *testing.T) {
	cfg := validConfig()
	cfg.Scope.LabelSelector = "not a valid selector!!"
	assertValidateError(t, cfg, "scope.labelSelector")
}

func TestValidate_GatingDefaults(t *testing.T) {
	cfg := validConfig()
	cfg.RevisionGating.RevisionHeaderName = ""
	cfg.RevisionGating.RevisionLabelKey = ""
	cfg.RevisionGating.RoleLabelKey = ""
	if err := cfg.Validate(); err != nil {
		t.Fatalf("empty protocol fields should default silently: %v", err)
	}
	if cfg.RevisionGating.RevisionHeaderName != DefaultRevisionHeader {
		t.Fatalf("want default %q, got %q", DefaultRevisionHeader, cfg.RevisionGating.RevisionHeaderName)
	}
	if cfg.RevisionGating.RevisionLabelKey != DefaultRevisionLabel {
		t.Fatalf("want default %q, got %q", DefaultRevisionLabel, cfg.RevisionGating.RevisionLabelKey)
	}
	if cfg.RevisionGating.RoleLabelKey != DefaultRoleLabel {
		t.Fatalf("want default %q, got %q", DefaultRoleLabel, cfg.RevisionGating.RoleLabelKey)
	}
}

func TestValidate_GatingRequiredRolesEmpty(t *testing.T) {
	cfg := validConfig()
	cfg.RevisionGating.RequiredRoles = nil
	assertValidateError(t, cfg, "revisionGating.requiredRoles")
}

func TestValidate_GatingRequiredRolesRejectsDuplicates(t *testing.T) {
	cfg := validConfig()
	cfg.RevisionGating.RequiredRoles = []string{"prefill", "prefill"}
	assertValidateError(t, cfg, "duplicate role")
}

func TestValidate_GatingRequiredRolesRejectsEmptyRole(t *testing.T) {
	cfg := validConfig()
	cfg.RevisionGating.RequiredRoles = []string{"prefill", ""}
	assertValidateError(t, cfg, "revisionGating.requiredRoles[1]")
}

func TestValidate_GatingRequired(t *testing.T) {
	cfg := validConfig()
	cfg.RevisionGating = nil
	assertValidateError(t, cfg, "revisionGating is required")
}

func TestValidate_GatingUnknownMode(t *testing.T) {
	cfg := validConfig()
	cfg.RevisionGating.Mode = "bogus"
	assertValidateError(t, cfg, "revisionGating.mode")
}

func TestValidate_GatingSumWithoutRequiredRoles(t *testing.T) {
	cfg := validConfig()
	cfg.RevisionGating.RequiredRoles = nil
	assertValidateError(t, cfg, "revisionGating.requiredRoles")
}

func TestValidate_GatingMaxRole(t *testing.T) {
	cfg := validConfig()
	cfg.RevisionGating.Mode = GatingModeMaxRole
	if err := cfg.Validate(); err != nil {
		t.Fatalf("gating.mode=max-role should validate: %v", err)
	}
	cfg.RevisionGating.RequiredRoles = nil
	assertValidateError(t, cfg, "revisionGating.requiredRoles")
}

func TestValidate_GatingDisabledSkipsSubValidation(t *testing.T) {
	cfg := validConfig()
	cfg.RevisionGating.Mode = GatingModeDisabled
	cfg.RevisionGating.RequiredRoles = nil
	if err := cfg.Validate(); err != nil {
		t.Fatalf("gating.mode=disabled should validate without requiredRoles: %v", err)
	}
}

func TestGating_Active(t *testing.T) {
	if (*RevisionGating)(nil).Active() {
		t.Errorf("nil should not be active")
	}
	sum := &RevisionGating{Mode: GatingModeSum, RequiredRoles: []string{"r"}}
	if !sum.Active() {
		t.Errorf("mode=sum with requiredRoles should be active")
	}
	sumMissingSub := &RevisionGating{Mode: GatingModeSum}
	if sumMissingSub.Active() {
		t.Errorf("mode=sum without requiredRoles should not be active")
	}
	maxRole := &RevisionGating{Mode: GatingModeMaxRole, RequiredRoles: []string{"r"}}
	if !maxRole.Active() {
		t.Errorf("mode=max-role with requiredRoles should be active")
	}
	maxRoleMissingSub := &RevisionGating{Mode: GatingModeMaxRole}
	if maxRoleMissingSub.Active() {
		t.Errorf("mode=max-role without requiredRoles should not be active")
	}
	disabled := &RevisionGating{Mode: GatingModeDisabled, RequiredRoles: []string{"r"}}
	if disabled.Active() {
		t.Errorf("mode=disabled should not be active")
	}
	unknown := &RevisionGating{Mode: "bogus"}
	if unknown.Active() {
		t.Errorf("unknown mode should not be active")
	}
}

func TestRevisionGating_UnmarshalJSON_LowercasesHeaderName(t *testing.T) {
	// Normalisation is at the JSON boundary; construction in Go leaves
	// the field alone.
	raw := []byte(`{"revisionHeaderName":"X-LLM-D-Disagg-Revision","mode":"sum","requiredRoles":["prefill","decode"]}`)
	var gating RevisionGating
	if err := json.Unmarshal(raw, &gating); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if gating.RevisionHeaderName != "x-llm-d-disagg-revision" {
		t.Fatalf("header not lowered on unmarshal: got %q", gating.RevisionHeaderName)
	}
	if got := strings.Join(gating.RequiredRoles, ","); got != "prefill,decode" {
		t.Fatalf("requiredRoles = %q, want prefill,decode", got)
	}
}

func TestRevisionGating_UnmarshalJSON_RejectsOldNestedRoles(t *testing.T) {
	raw := []byte(`{"mode":"sum","requireRoles":{"values":["prefill","decode"]}}`)
	var gating RevisionGating
	if err := json.Unmarshal(raw, &gating); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("want unknown field error, got %v", err)
	}
}

func TestValidate_LeavesHeaderNameAloneOnInCodeConstruction(t *testing.T) {
	// Validate must not mutate; normalisation is at unmarshal time.
	cfg := validConfig()
	cfg.RevisionGating.RevisionHeaderName = "X-LLM-D-Disagg-Revision"
	if err := cfg.Validate(); err != nil {
		t.Fatalf("mixed-case header name should validate: %v", err)
	}
	if cfg.RevisionGating.RevisionHeaderName != "X-LLM-D-Disagg-Revision" {
		t.Fatalf("Validate mutated RevisionHeaderName: got %q", cfg.RevisionGating.RevisionHeaderName)
	}
}

func assertValidateError(t *testing.T, cfg Config, wantSubstring string) {
	t.Helper()
	err := cfg.Validate()
	if err == nil {
		t.Fatalf("expected error containing %q, got nil", wantSubstring)
	}
	if !strings.Contains(err.Error(), wantSubstring) {
		t.Fatalf("error %q does not contain %q", err.Error(), wantSubstring)
	}
}
