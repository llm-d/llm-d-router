package disaggregation

import (
	"context"
	"testing"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/picker/random"
	singleprofilehandler "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/profilehandler/single"
	"github.com/llm-d/llm-d-router/pkg/epp/requestcontrol"
	"github.com/llm-d/llm-d-router/pkg/epp/scheduling"
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

// TestWireInto_GatingAtHead_PreferAtTail proves the placement contract:
// the revision decision is made at the head (gating first, then strict —
// only one of them fires per request), any operator-declared filters run
// on the resulting single-revision pool, and prefer runs at the tail so
// its keep-if-empty fallback sees the fully-narrowed pool.
func TestWireInto_GatingAtHead_PreferAtTail(t *testing.T) {
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
	controller := newTestController(config)
	requestControlConfig := requestcontrol.NewConfig()

	if err := WireInto(schedulerConfig, requestControlConfig, controller); err != nil {
		t.Fatalf("WireInto: %v", err)
	}

	// Read back the filter chain via profile.String() — the framework does
	// not expose the private slice directly. Expected order in the string:
	// gating, strict, operator, prefer.
	got := profile.String()
	gatingAt := indexOf(got, gatingFilterType)
	strictAt := indexOf(got, strictFilterType)
	operatorAt := indexOf(got, "operator-filter")
	preferAt := indexOf(got, preferFilterType)
	if gatingAt < 0 || strictAt < 0 || operatorAt < 0 || preferAt < 0 {
		t.Fatalf("expected all four filters in profile string, got %q", got)
	}
	if gatingAt >= strictAt || strictAt >= operatorAt || operatorAt >= preferAt {
		t.Fatalf("expected order gating < strict < operator < prefer in %q (positions %d, %d, %d, %d)",
			got, gatingAt, strictAt, operatorAt, preferAt)
	}
}

func TestWireInto_OmitsStrictWhenNoStrictSelectors(t *testing.T) {
	schedulerConfig := buildSchedulerConfigForTest(t)
	profile := schedulerConfig.Profiles()["default"]

	config := validConfig()
	config.Selectors[0].Mode = ModePrefer
	controller := newTestController(config)
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
