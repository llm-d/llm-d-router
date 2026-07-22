package disaggregation

import (
	"encoding/json"
	"strings"
	"testing"
)

func validConfig() Config {
	return Config{
		Enabled: true,
		Scope: Scope{
			LabelSelector: "disaggregatedset.x-k8s.io/name=my-set",
		},
		Selectors: []Selector{
			{
				Name:       "revision",
				HeaderName: "x-disagg-revision",
				LabelKey:   "disaggregatedset.x-k8s.io/revision",
				Mode:       ModeStrict,
			},
		},
		Gating: &Gating{
			Mode: GatingModeSum,
			RequireRoles: &RequireRoles{
				LabelKey: "disaggregatedset.x-k8s.io/role",
				Values:   []string{"prefill", "decode"},
			},
		},
	}
}

func TestValidate_Valid(t *testing.T) {
	cfg := validConfig()
	if err := cfg.Validate(); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
}

func TestValidate_Disabled(t *testing.T) {
	cfg := Config{Enabled: false}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("disabled config should validate trivially: %v", err)
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

func TestValidate_EmptySelectors(t *testing.T) {
	cfg := validConfig()
	cfg.Selectors = nil
	assertValidateError(t, cfg, "selectors")
}

func TestValidate_SelectorMissingName(t *testing.T) {
	cfg := validConfig()
	cfg.Selectors[0].Name = ""
	assertValidateError(t, cfg, "selectors[0].name")
}

func TestValidate_SelectorMissingHeaderName(t *testing.T) {
	cfg := validConfig()
	cfg.Selectors[0].HeaderName = ""
	assertValidateError(t, cfg, "selectors[0].headerName")
}

func TestValidate_SelectorMissingLabelKey(t *testing.T) {
	cfg := validConfig()
	cfg.Selectors[0].LabelKey = ""
	assertValidateError(t, cfg, "selectors[0].labelKey")
}

func TestValidate_SelectorUnknownMode(t *testing.T) {
	cfg := validConfig()
	cfg.Selectors[0].Mode = "loose"
	assertValidateError(t, cfg, "selectors[0].mode")
}

func TestValidate_DuplicateSelectorName(t *testing.T) {
	cfg := validConfig()
	cfg.Selectors = append(cfg.Selectors, Selector{
		Name:       "revision", // duplicate
		HeaderName: "x-disagg-other",
		LabelKey:   "foo",
		Mode:       ModeStrict,
	})
	assertValidateError(t, cfg, "duplicate selector name")
}

func TestValidate_DuplicateHeaderName(t *testing.T) {
	cfg := validConfig()
	cfg.Selectors = append(cfg.Selectors, Selector{
		Name:       "other",
		HeaderName: "x-disagg-revision", // duplicate
		LabelKey:   "foo",
		Mode:       ModeStrict,
	})
	assertValidateError(t, cfg, "duplicate header name")
}

func TestValidate_GatingRequireRolesMissingLabelKey(t *testing.T) {
	cfg := validConfig()
	cfg.Gating.RequireRoles.LabelKey = ""
	assertValidateError(t, cfg, "gating.requireRoles.labelKey")
}

func TestValidate_GatingRequireRolesEmptyValues(t *testing.T) {
	cfg := validConfig()
	cfg.Gating.RequireRoles.Values = nil
	assertValidateError(t, cfg, "gating.requireRoles.values")
}

func TestValidate_GatingOptional(t *testing.T) {
	// The whole gating block is optional; leaving it nil validates and
	// simply skips wiring the gating filter at boot.
	cfg := validConfig()
	cfg.Gating = nil
	if err := cfg.Validate(); err != nil {
		t.Fatalf("config without gating should validate: %v", err)
	}
}

func TestValidate_GatingUnknownMode(t *testing.T) {
	cfg := validConfig()
	cfg.Gating.Mode = "bogus"
	assertValidateError(t, cfg, "gating.mode")
}

func TestValidate_GatingSumWithoutRequireRoles(t *testing.T) {
	// mode=sum needs the requireRoles sub-block. Missing it is a config
	// error, not a silent no-op.
	cfg := validConfig()
	cfg.Gating.RequireRoles = nil
	assertValidateError(t, cfg, "gating.requireRoles is required")
}

func TestValidate_GatingDisabledSkipsSubValidation(t *testing.T) {
	// mode=disabled is a legitimate way to keep the block for documentation
	// while turning the filter off; the sub-block does not need to be set.
	cfg := validConfig()
	cfg.Gating.Mode = GatingModeDisabled
	cfg.Gating.RequireRoles = nil
	if err := cfg.Validate(); err != nil {
		t.Fatalf("gating.mode=disabled should validate without requireRoles: %v", err)
	}
}

func TestGating_Active(t *testing.T) {
	if (*Gating)(nil).Active() {
		t.Errorf("nil should not be active")
	}
	sum := &Gating{Mode: GatingModeSum, RequireRoles: &RequireRoles{LabelKey: "k", Values: []string{"r"}}}
	if !sum.Active() {
		t.Errorf("mode=sum with requireRoles should be active")
	}
	sumMissingSub := &Gating{Mode: GatingModeSum}
	if sumMissingSub.Active() {
		t.Errorf("mode=sum without requireRoles should not be active")
	}
	disabled := &Gating{Mode: GatingModeDisabled, RequireRoles: &RequireRoles{LabelKey: "k", Values: []string{"r"}}}
	if disabled.Active() {
		t.Errorf("mode=disabled should not be active")
	}
	unknown := &Gating{Mode: "bogus"}
	if unknown.Active() {
		t.Errorf("unknown mode should not be active")
	}
}

func TestSelector_UnmarshalJSON_LowercasesHeaderName(t *testing.T) {
	// Normalisation is at the JSON boundary now; construction in Go leaves
	// the field alone.
	raw := []byte(`{"name":"revision","headerName":"X-Disagg-Revision","labelKey":"lk","mode":"strict"}`)
	var selector Selector
	if err := json.Unmarshal(raw, &selector); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if selector.HeaderName != "x-disagg-revision" {
		t.Fatalf("header not lowered on unmarshal: got %q", selector.HeaderName)
	}
}

func TestValidate_LeavesHeaderNameAloneOnInCodeConstruction(t *testing.T) {
	// Validate must not mutate — normalisation is at unmarshal time.
	cfg := validConfig()
	cfg.Selectors[0].HeaderName = "X-Disagg-Revision"
	if err := cfg.Validate(); err != nil {
		t.Fatalf("mixed-case header name should validate: %v", err)
	}
	if cfg.Selectors[0].HeaderName != "X-Disagg-Revision" {
		t.Fatalf("Validate mutated HeaderName: got %q", cfg.Selectors[0].HeaderName)
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
