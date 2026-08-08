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

package disagg

import (
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"
	"github.com/stretchr/testify/require"
)

const (
	testMetricModelName = "test-model"
	testPluginName      = "test-plugin"
	testPluginType      = "test-type"
)

func TestRegisterMetrics(t *testing.T) {
	registry := prometheus.NewRegistry()

	require.NoError(t, registerMetrics(registry))
	require.NoError(t, registerMetrics(registry))
}

func TestSchedulerPDDecisionCount(t *testing.T) {
	SchedulerPDDecisionCount.Reset()
	LlmdPDDecisionCount.Reset()

	model := testMetricModelName

	RecordPDDecision(testPluginName, testPluginType, model, DecisionTypePrefillDecode)
	RecordPDDecision(testPluginName, testPluginType, model, DecisionTypeDecodeOnly)
	RecordPDDecision(testPluginName, testPluginType, model, DecisionTypePrefillDecode)

	expected := `
		# HELP llm_d_inference_scheduler_pd_decision_total [ALPHA] [Deprecated: Use llm_d_epp_pd_decision_total] Total number of P/D disaggregation decisions made
		# TYPE llm_d_inference_scheduler_pd_decision_total counter
		llm_d_inference_scheduler_pd_decision_total{decision_type="decode-only",model_name="test-model"} 1
		llm_d_inference_scheduler_pd_decision_total{decision_type="prefill-decode",model_name="test-model"} 2
	`

	if err := testutil.CollectAndCompare(SchedulerPDDecisionCount, strings.NewReader(expected),
		"llm_d_inference_scheduler_pd_decision_total"); err != nil {
		t.Errorf("RecordPDDecision() failed: %v", err)
	}

	expectedNew := `
		# HELP llm_d_epp_pd_decision_total [ALPHA] Total number of P/D disaggregation decisions made
		# TYPE llm_d_epp_pd_decision_total counter
		llm_d_epp_pd_decision_total{decision_type="decode-only",model_name="test-model",plugin_name="test-plugin",plugin_type="test-type"} 1
		llm_d_epp_pd_decision_total{decision_type="prefill-decode",model_name="test-model",plugin_name="test-plugin",plugin_type="test-type"} 2
	`

	if err := testutil.CollectAndCompare(LlmdPDDecisionCount, strings.NewReader(expectedNew),
		"llm_d_epp_pd_decision_total"); err != nil {
		t.Errorf("RecordPDDecision() new failed: %v", err)
	}
}

func TestRecordDisaggDecision(t *testing.T) {
	// Reset the counters before the test to avoid interference from other tests.
	SchedulerDisaggDecisionCount.Reset()
	LlmdDisaggDecisionCount.Reset()

	model := testMetricModelName
	RecordDisaggDecision(testPluginName, testPluginType, model, DecisionTypeDecodeOnly)
	RecordDisaggDecision(testPluginName, testPluginType, model, DecisionTypePrefillDecode)
	RecordDisaggDecision(testPluginName, testPluginType, model, DecisionTypePrefillDecode)
	RecordDisaggDecision(testPluginName, testPluginType, model, DecisionTypeEncodeDecode)
	RecordDisaggDecision(testPluginName, testPluginType, model, DecisionTypeEncodePrefillDecode)
	RecordDisaggDecision(testPluginName, testPluginType, model, DecisionTypeEncodePrefillDecode)
	RecordDisaggDecision(testPluginName, testPluginType, model, DecisionTypeEncodePrefillDecode)

	expected := `
		# HELP llm_d_inference_scheduler_disagg_decision_total [ALPHA] [Deprecated: Use llm_d_epp_disagg_decision_total] Total number of disaggregation routing decisions made
		# TYPE llm_d_inference_scheduler_disagg_decision_total counter
		llm_d_inference_scheduler_disagg_decision_total{decision_type="decode-only",model_name="test-model"} 1
		llm_d_inference_scheduler_disagg_decision_total{decision_type="encode-decode",model_name="test-model"} 1
		llm_d_inference_scheduler_disagg_decision_total{decision_type="encode-prefill-decode",model_name="test-model"} 3
		llm_d_inference_scheduler_disagg_decision_total{decision_type="prefill-decode",model_name="test-model"} 2
	`

	if err := testutil.CollectAndCompare(SchedulerDisaggDecisionCount, strings.NewReader(expected),
		"llm_d_inference_scheduler_disagg_decision_total"); err != nil {
		t.Errorf("RecordDisaggDecision() failed: %v", err)
	}

	expectedNew := `
		# HELP llm_d_epp_disagg_decision_total [ALPHA] Total number of disaggregation routing decisions made
		# TYPE llm_d_epp_disagg_decision_total counter
		llm_d_epp_disagg_decision_total{decision_type="decode-only",model_name="test-model",plugin_name="test-plugin",plugin_type="test-type"} 1
		llm_d_epp_disagg_decision_total{decision_type="encode-decode",model_name="test-model",plugin_name="test-plugin",plugin_type="test-type"} 1
		llm_d_epp_disagg_decision_total{decision_type="encode-prefill-decode",model_name="test-model",plugin_name="test-plugin",plugin_type="test-type"} 3
		llm_d_epp_disagg_decision_total{decision_type="prefill-decode",model_name="test-model",plugin_name="test-plugin",plugin_type="test-type"} 2
	`

	if err := testutil.CollectAndCompare(LlmdDisaggDecisionCount, strings.NewReader(expectedNew),
		"llm_d_epp_disagg_decision_total"); err != nil {
		t.Errorf("RecordDisaggDecision() new failed: %v", err)
	}
}

func TestRecordDisaggDecisionEmptyModel(t *testing.T) {
	SchedulerDisaggDecisionCount.Reset()
	LlmdDisaggDecisionCount.Reset()

	RecordDisaggDecision(testPluginName, testPluginType, "", DecisionTypeDecodeOnly)

	expected := `
		# HELP llm_d_inference_scheduler_disagg_decision_total [ALPHA] [Deprecated: Use llm_d_epp_disagg_decision_total] Total number of disaggregation routing decisions made
		# TYPE llm_d_inference_scheduler_disagg_decision_total counter
		llm_d_inference_scheduler_disagg_decision_total{decision_type="decode-only",model_name="unknown"} 1
	`

	if err := testutil.CollectAndCompare(SchedulerDisaggDecisionCount, strings.NewReader(expected),
		"llm_d_inference_scheduler_disagg_decision_total"); err != nil {
		t.Errorf("RecordDisaggDecision() with empty model failed: %v", err)
	}

	expectedNew := `
		# HELP llm_d_epp_disagg_decision_total [ALPHA] Total number of disaggregation routing decisions made
		# TYPE llm_d_epp_disagg_decision_total counter
		llm_d_epp_disagg_decision_total{decision_type="decode-only",model_name="unknown",plugin_name="test-plugin",plugin_type="test-type"} 1
	`

	if err := testutil.CollectAndCompare(LlmdDisaggDecisionCount, strings.NewReader(expectedNew),
		"llm_d_epp_disagg_decision_total"); err != nil {
		t.Errorf("RecordDisaggDecision() new empty model failed: %v", err)
	}
}

func TestDisaggDecisionType(t *testing.T) {
	tests := []struct {
		encodeUsed  bool
		prefillUsed bool
		want        string
	}{
		{false, false, DecisionTypeDecodeOnly},
		{false, true, DecisionTypePrefillDecode},
		{true, false, DecisionTypeEncodeDecode},
		{true, true, DecisionTypeEncodePrefillDecode},
	}
	for _, tt := range tests {
		got := DisaggDecisionType(tt.encodeUsed, tt.prefillUsed)
		if got != tt.want {
			t.Errorf("DisaggDecisionType(%v, %v) = %q, want %q", tt.encodeUsed, tt.prefillUsed, got, tt.want)
		}
	}
}

func TestRecordDeciderEvaluation(t *testing.T) {
	llmdDeciderEvaluationCount.Reset()

	model := testMetricModelName
	decider := "prefix-based-pd-decider"

	recordDeciderEvaluation(model, decider, deciderReasonDisabled)
	recordDeciderEvaluation(model, decider, deciderReasonInputTooShort)
	recordDeciderEvaluation(model, decider, deciderReasonInputTooShort)
	recordDeciderEvaluation(model, decider, deciderReasonSuffixCached)
	recordDeciderEvaluation(model, decider, deciderReasonError)
	recordDeciderEvaluation(model, decider, deciderReasonDisaggregated)
	recordDeciderEvaluation(model, decider, deciderReasonDisaggregated)
	recordDeciderEvaluation(model, decider, deciderReasonDisaggregated)

	expected := `
		# HELP llm_d_epp_decider_evaluation_total [ALPHA] Total number of disaggregation decider evaluations by reason
		# TYPE llm_d_epp_decider_evaluation_total counter
		llm_d_epp_decider_evaluation_total{decider="prefix-based-pd-decider",model_name="test-model",reason="disabled"} 1
		llm_d_epp_decider_evaluation_total{decider="prefix-based-pd-decider",model_name="test-model",reason="disaggregated"} 3
		llm_d_epp_decider_evaluation_total{decider="prefix-based-pd-decider",model_name="test-model",reason="error"} 1
		llm_d_epp_decider_evaluation_total{decider="prefix-based-pd-decider",model_name="test-model",reason="input_too_short"} 2
		llm_d_epp_decider_evaluation_total{decider="prefix-based-pd-decider",model_name="test-model",reason="suffix_cached"} 1
	`

	if err := testutil.CollectAndCompare(llmdDeciderEvaluationCount, strings.NewReader(expected),
		"llm_d_epp_decider_evaluation_total"); err != nil {
		t.Errorf("recordDeciderEvaluation() failed: %v", err)
	}
}

func TestRecordDeciderEvaluationEmptyModel(t *testing.T) {
	llmdDeciderEvaluationCount.Reset()

	recordDeciderEvaluation("", "my-decider", deciderReasonDisaggregated)

	expected := `
		# HELP llm_d_epp_decider_evaluation_total [ALPHA] Total number of disaggregation decider evaluations by reason
		# TYPE llm_d_epp_decider_evaluation_total counter
		llm_d_epp_decider_evaluation_total{decider="my-decider",model_name="unknown",reason="disaggregated"} 1
	`

	if err := testutil.CollectAndCompare(llmdDeciderEvaluationCount, strings.NewReader(expected),
		"llm_d_epp_decider_evaluation_total"); err != nil {
		t.Errorf("recordDeciderEvaluation() with empty model failed: %v", err)
	}
}
