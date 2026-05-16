package metrics

import (
	"strings"
	"testing"

	"github.com/prometheus/client_golang/prometheus/testutil"
)

func TestSchedulerPDDecisionCount(t *testing.T) {
	model := "test-model"

	RecordPDDecision(model, DecisionTypePrefillDecode)
	RecordPDDecision(model, DecisionTypeDecodeOnly)
	RecordPDDecision(model, DecisionTypePrefillDecode)
	expected := `
		# HELP llm_d_inference_scheduler_pd_decision_total [ALPHA] Total number of P/D disaggregation decisions made
		# TYPE llm_d_inference_scheduler_pd_decision_total counter
		llm_d_inference_scheduler_pd_decision_total{decision_type="decode-only",model_name="test-model"} 1
		llm_d_inference_scheduler_pd_decision_total{decision_type="prefill-decode",model_name="test-model"} 2
	`

	if err := testutil.CollectAndCompare(SchedulerPDDecisionCount, strings.NewReader(expected),
		"llm_d_inference_scheduler_pd_decision_total"); err != nil {
		t.Errorf("RecordPDDecision() failed: %v", err)
	}
}

func TestRecordDisaggDecision(t *testing.T) {
	// Reset the counter before the test to avoid interference from other tests.
	SchedulerDisaggDecisionCount.Reset()

	model := "test-model"
	RecordDisaggDecision(model, DecisionTypeDecodeOnly)
	RecordDisaggDecision(model, DecisionTypePrefillDecode)
	RecordDisaggDecision(model, DecisionTypePrefillDecode)
	RecordDisaggDecision(model, DecisionTypeEncodeDecode)
	RecordDisaggDecision(model, DecisionTypeEncodePrefillDecode)
	RecordDisaggDecision(model, DecisionTypeEncodePrefillDecode)
	RecordDisaggDecision(model, DecisionTypeEncodePrefillDecode)

	expected := `
		# HELP llm_d_inference_scheduler_disagg_decision_total [ALPHA] Total number of disaggregation routing decisions made
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
}

func TestRecordDisaggDecisionEmptyModel(t *testing.T) {
	SchedulerDisaggDecisionCount.Reset()

	RecordDisaggDecision("", DecisionTypeDecodeOnly)

	expected := `
		# HELP llm_d_inference_scheduler_disagg_decision_total [ALPHA] Total number of disaggregation routing decisions made
		# TYPE llm_d_inference_scheduler_disagg_decision_total counter
		llm_d_inference_scheduler_disagg_decision_total{decision_type="decode-only",model_name="unknown"} 1
	`

	if err := testutil.CollectAndCompare(SchedulerDisaggDecisionCount, strings.NewReader(expected),
		"llm_d_inference_scheduler_disagg_decision_total"); err != nil {
		t.Errorf("RecordDisaggDecision() with empty model failed: %v", err)
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
	SchedulerDeciderEvaluationCount.Reset()

	model := "test-model"
	decider := "prefix-based-pd-decider"

	RecordDeciderEvaluation(model, decider, DeciderReasonDisabled)
	RecordDeciderEvaluation(model, decider, DeciderReasonInputTooShort)
	RecordDeciderEvaluation(model, decider, DeciderReasonInputTooShort)
	RecordDeciderEvaluation(model, decider, DeciderReasonSuffixCached)
	RecordDeciderEvaluation(model, decider, DeciderReasonError)
	RecordDeciderEvaluation(model, decider, DeciderReasonDisaggregated)
	RecordDeciderEvaluation(model, decider, DeciderReasonDisaggregated)
	RecordDeciderEvaluation(model, decider, DeciderReasonDisaggregated)

	expected := `
		# HELP llm_d_inference_scheduler_decider_evaluation_total [ALPHA] Total number of disaggregation decider evaluations by reason
		# TYPE llm_d_inference_scheduler_decider_evaluation_total counter
		llm_d_inference_scheduler_decider_evaluation_total{decider="prefix-based-pd-decider",model_name="test-model",reason="disabled"} 1
		llm_d_inference_scheduler_decider_evaluation_total{decider="prefix-based-pd-decider",model_name="test-model",reason="disaggregated"} 3
		llm_d_inference_scheduler_decider_evaluation_total{decider="prefix-based-pd-decider",model_name="test-model",reason="error"} 1
		llm_d_inference_scheduler_decider_evaluation_total{decider="prefix-based-pd-decider",model_name="test-model",reason="input_too_short"} 2
		llm_d_inference_scheduler_decider_evaluation_total{decider="prefix-based-pd-decider",model_name="test-model",reason="suffix_cached"} 1
	`

	if err := testutil.CollectAndCompare(SchedulerDeciderEvaluationCount, strings.NewReader(expected),
		"llm_d_inference_scheduler_decider_evaluation_total"); err != nil {
		t.Errorf("RecordDeciderEvaluation() failed: %v", err)
	}
}

func TestRecordDeciderEvaluationEmptyModel(t *testing.T) {
	SchedulerDeciderEvaluationCount.Reset()

	RecordDeciderEvaluation("", "my-decider", DeciderReasonDisaggregated)

	expected := `
		# HELP llm_d_inference_scheduler_decider_evaluation_total [ALPHA] Total number of disaggregation decider evaluations by reason
		# TYPE llm_d_inference_scheduler_decider_evaluation_total counter
		llm_d_inference_scheduler_decider_evaluation_total{decider="my-decider",model_name="unknown",reason="disaggregated"} 1
	`

	if err := testutil.CollectAndCompare(SchedulerDeciderEvaluationCount, strings.NewReader(expected),
		"llm_d_inference_scheduler_decider_evaluation_total"); err != nil {
		t.Errorf("RecordDeciderEvaluation() with empty model failed: %v", err)
	}
}
