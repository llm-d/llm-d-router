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
		HeaderSelectors: []HeaderSelector{
			{
				Name:       "revision",
				HeaderName: "x-disagg-revision",
				LabelKey:   "disaggregatedset.x-k8s.io/revision",
				Mode:       ModeStrict,
			},
		},
		RevisionGating: &RevisionGating{
			RevisionLabelKey: "disaggregatedset.x-k8s.io/revision",
			RoleLabelKey:     "disaggregatedset.x-k8s.io/role",
			Mode:             GatingModeSum,
			RequireRoles:     &RequireRoles{Values: []string{"prefill", "decode"}},
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

func TestValidate_EmptyHeaderSelectorsIsAllowed(t *testing.T) {
	// Gating-only configs (no header pinning) are legitimate because the
	// operator wants server-side revision shaping without a client-
	// controlled pin. HeaderSelectors is optional.
	cfg := validConfig()
	cfg.HeaderSelectors = nil
	if err := cfg.Validate(); err != nil {
		t.Fatalf("empty headerSelectors should be allowed: %v", err)
	}
}

func TestValidate_SelectorMissingName(t *testing.T) {
	cfg := validConfig()
	cfg.HeaderSelectors[0].Name = ""
	assertValidateError(t, cfg, "headerSelectors[0].name")
}

func TestValidate_SelectorMissingHeaderName(t *testing.T) {
	cfg := validConfig()
	cfg.HeaderSelectors[0].HeaderName = ""
	assertValidateError(t, cfg, "headerSelectors[0].headerName")
}

func TestValidate_SelectorMissingLabelKey(t *testing.T) {
	cfg := validConfig()
	cfg.HeaderSelectors[0].LabelKey = ""
	assertValidateError(t, cfg, "headerSelectors[0].labelKey")
}

func TestValidate_SelectorUnknownMode(t *testing.T) {
	cfg := validConfig()
	cfg.HeaderSelectors[0].Mode = "loose"
	assertValidateError(t, cfg, "headerSelectors[0].mode")
}

func TestValidate_DuplicateSelectorName(t *testing.T) {
	cfg := validConfig()
	cfg.HeaderSelectors = append(cfg.HeaderSelectors, HeaderSelector{
		Name:       "revision", // duplicate
		HeaderName: "x-disagg-other",
		LabelKey:   "foo",
		Mode:       ModeStrict,
	})
	assertValidateError(t, cfg, "duplicate name")
}

func TestValidate_DuplicateHeaderName(t *testing.T) {
	cfg := validConfig()
	cfg.HeaderSelectors = append(cfg.HeaderSelectors, HeaderSelector{
		Name:       "other",
		HeaderName: "x-disagg-revision", // duplicate
		LabelKey:   "foo",
		Mode:       ModeStrict,
	})
	assertValidateError(t, cfg, "duplicate headerName")
}

func TestValidate_GatingRoleLabelKeyDefaults(t *testing.T) {
	cfg := validConfig()
	cfg.RevisionGating.RoleLabelKey = ""
	if err := cfg.Validate(); err != nil {
		t.Fatalf("empty roleLabelKey should default silently: %v", err)
	}
	if cfg.RevisionGating.RoleLabelKey != DefaultRoleLabel {
		t.Fatalf("want default %q, got %q", DefaultRoleLabel, cfg.RevisionGating.RoleLabelKey)
	}
}

func TestValidate_GatingRevisionLabelKeyDefaults(t *testing.T) {
	cfg := validConfig()
	cfg.RevisionGating.RevisionLabelKey = ""
	if err := cfg.Validate(); err != nil {
		t.Fatalf("empty revisionLabelKey should default silently: %v", err)
	}
	if cfg.RevisionGating.RevisionLabelKey != DefaultRevisionLabel {
		t.Fatalf("want default %q, got %q", DefaultRevisionLabel, cfg.RevisionGating.RevisionLabelKey)
	}
}

func TestValidate_GatingRequireRolesEmptyValues(t *testing.T) {
	cfg := validConfig()
	cfg.RevisionGating.RequireRoles.Values = nil
	assertValidateError(t, cfg, "revisionGating.requireRoles.values")
}

func TestValidate_GatingRequireRolesRejectsDuplicates(t *testing.T) {
	cfg := validConfig()
	cfg.RevisionGating.RequireRoles.Values = []string{"prefill", "prefill"}
	assertValidateError(t, cfg, "duplicate role")
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

func TestValidate_GatingSumWithoutRequireRoles(t *testing.T) {
	// mode=sum needs the requireRoles sub-block. Missing it is a config
	// error, not a silent no-op.
	cfg := validConfig()
	cfg.RevisionGating.RequireRoles = nil
	assertValidateError(t, cfg, "revisionGating.requireRoles is required")
}

func TestValidate_GatingMaxRole(t *testing.T) {
	cfg := validConfig()
	cfg.RevisionGating.Mode = GatingModeMaxRole
	if err := cfg.Validate(); err != nil {
		t.Fatalf("gating.mode=max-role should validate: %v", err)
	}
	cfg.RevisionGating.RequireRoles = nil
	assertValidateError(t, cfg, "revisionGating.requireRoles is required")
}

func TestValidate_GatingDisabledSkipsSubValidation(t *testing.T) {
	// mode=disabled is a legitimate way to keep the block for documentation
	// while turning gating off; the sub-block does not need to be set.
	cfg := validConfig()
	cfg.RevisionGating.Mode = GatingModeDisabled
	cfg.RevisionGating.RequireRoles = nil
	if err := cfg.Validate(); err != nil {
		t.Fatalf("gating.mode=disabled should validate without requireRoles: %v", err)
	}
}

func TestGating_Active(t *testing.T) {
	if (*RevisionGating)(nil).Active() {
		t.Errorf("nil should not be active")
	}
	sum := &RevisionGating{Mode: GatingModeSum, RequireRoles: &RequireRoles{Values: []string{"r"}}}
	if !sum.Active() {
		t.Errorf("mode=sum with requireRoles should be active")
	}
	sumMissingSub := &RevisionGating{Mode: GatingModeSum}
	if sumMissingSub.Active() {
		t.Errorf("mode=sum without requireRoles should not be active")
	}
	maxRole := &RevisionGating{Mode: GatingModeMaxRole, RequireRoles: &RequireRoles{Values: []string{"r"}}}
	if !maxRole.Active() {
		t.Errorf("mode=max-role with requireRoles should be active")
	}
	maxRoleMissingSub := &RevisionGating{Mode: GatingModeMaxRole}
	if maxRoleMissingSub.Active() {
		t.Errorf("mode=max-role without requireRoles should not be active")
	}
	disabled := &RevisionGating{Mode: GatingModeDisabled, RequireRoles: &RequireRoles{Values: []string{"r"}}}
	if disabled.Active() {
		t.Errorf("mode=disabled should not be active")
	}
	unknown := &RevisionGating{Mode: "bogus"}
	if unknown.Active() {
		t.Errorf("unknown mode should not be active")
	}
}

func TestSelector_UnmarshalJSON_LowercasesHeaderName(t *testing.T) {
	// Normalisation is at the JSON boundary; construction in Go leaves
	// the field alone.
	raw := []byte(`{"name":"revision","headerName":"X-Disagg-Revision","labelKey":"lk","mode":"strict"}`)
	var selector HeaderSelector
	if err := json.Unmarshal(raw, &selector); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if selector.HeaderName != "x-disagg-revision" {
		t.Fatalf("header not lowered on unmarshal: got %q", selector.HeaderName)
	}
}

func TestSelector_UnmarshalJSON_RejectsUnknownFields(t *testing.T) {
	raw := []byte(`{"name":"revision","headerName":"x-disagg-revision","labelKey":"lk","mode":"strict","unknown":"value"}`)
	var selector HeaderSelector
	if err := json.Unmarshal(raw, &selector); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("want unknown field error, got %v", err)
	}
}

func TestValidate_LeavesHeaderNameAloneOnInCodeConstruction(t *testing.T) {
	// Validate must not mutate; normalisation is at unmarshal time.
	cfg := validConfig()
	cfg.HeaderSelectors[0].HeaderName = "X-Disagg-Revision"
	if err := cfg.Validate(); err != nil {
		t.Fatalf("mixed-case header name should validate: %v", err)
	}
	if cfg.HeaderSelectors[0].HeaderName != "X-Disagg-Revision" {
		t.Fatalf("Validate mutated HeaderName: got %q", cfg.HeaderSelectors[0].HeaderName)
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
