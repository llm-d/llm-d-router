package disaggregation

import (
	"context"
	"testing"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"

	singleprofilehandler "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/random"
)

// buildSchedulerConfigForTest builds a minimal scheduler config with a
// single "default" profile and a random picker, which is the shape produced
// by the loader for our helm values files.
func buildSchedulerConfigForTest(t *testing.T) *scheduling.SchedulerConfig {
	t.Helper()
	profile := scheduling.NewSchedulerProfile()
	profile.WithPicker(random.NewRandomPicker(1))
	profiles := map[string]fwksched.SchedulerProfile{"default": profile}
	return scheduling.NewSchedulerConfig(singleprofilehandler.NewSingleProfileHandler(), profiles)
}

// TestWireInto_StrictAtHead_PreferAtTail proves the placement contract: an
// existing operator-declared filter ends up sandwiched between our strict and
// prefer filters.
func TestWireInto_StrictAtHead_PreferAtTail(t *testing.T) {
	schedulerConfig := buildSchedulerConfigForTest(t)
	profile := schedulerConfig.Profiles()["default"]
	profile.AppendFilter(&namedTestFilter{name: "operator-filter"})

	config := validConfig()
	config.Selectors = append(config.Selectors, Selector{
		Name:       "slice",
		HeaderName: "x-disagg-slice",
		LabelKey:   "mistral.ai/slice",
		Mode:       ModePrefer,
	})
	controller := NewController(config, nil)
	requestControlConfig := requestcontrol.NewConfig()

	if err := WireInto(schedulerConfig, requestControlConfig, controller); err != nil {
		t.Fatalf("WireInto: %v", err)
	}

	// Read back the filter chain via profile.String() — the framework does
	// not expose the private slice directly. Expected order in the string:
	// strict, operator, prefer, gating.
	got := profile.String()
	strictAt := indexOf(got, strictFilterType)
	operatorAt := indexOf(got, "operator-filter")
	preferAt := indexOf(got, preferFilterType)
	gatingAt := indexOf(got, gatingFilterType)
	if strictAt < 0 || operatorAt < 0 || preferAt < 0 || gatingAt < 0 {
		t.Fatalf("expected all four filters in profile string, got %q", got)
	}
	if !(strictAt < operatorAt && operatorAt < preferAt && preferAt < gatingAt) {
		t.Fatalf("expected order strict < operator < prefer < gating in %q (positions %d, %d, %d, %d)",
			got, strictAt, operatorAt, preferAt, gatingAt)
	}
}

func TestWireInto_OmitsStrictWhenNoStrictSelectors(t *testing.T) {
	schedulerConfig := buildSchedulerConfigForTest(t)
	profile := schedulerConfig.Profiles()["default"]

	config := validConfig()
	config.Selectors[0].Mode = ModePrefer
	controller := NewController(config, nil)
	requestControlConfig := requestcontrol.NewConfig()

	if err := WireInto(schedulerConfig, requestControlConfig, controller); err != nil {
		t.Fatalf("WireInto: %v", err)
	}
	got := profile.String()
	if indexOf(got, strictFilterType) >= 0 {
		t.Errorf("no strict selectors → no strict filter registered; got %q", got)
	}
	if indexOf(got, preferFilterType) < 0 {
		t.Errorf("prefer selector present → prefer filter expected; got %q", got)
	}
}

func TestWireInto_NilControllerIsNoop(t *testing.T) {
	schedulerConfig := buildSchedulerConfigForTest(t)
	requestControlConfig := requestcontrol.NewConfig()
	if err := WireInto(schedulerConfig, requestControlConfig, nil); err != nil {
		t.Fatalf("nil controller: got %v, want nil", err)
	}
}

// --- helpers ----------------------------------------------------------------

type namedTestFilter struct {
	name string
}

func (f *namedTestFilter) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Type: f.name, Name: f.name}
}
func (f *namedTestFilter) Filter(_ context.Context, _ *fwksched.InferenceRequest, pods []fwksched.Endpoint) []fwksched.Endpoint {
	return pods
}

func indexOf(haystack, needle string) int {
	for offset := 0; offset+len(needle) <= len(haystack); offset++ {
		if haystack[offset:offset+len(needle)] == needle {
			return offset
		}
	}
	return -1
}
