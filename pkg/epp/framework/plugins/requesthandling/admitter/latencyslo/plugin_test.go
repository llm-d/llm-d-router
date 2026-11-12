/*
Copyright 2025 The Kubernetes Authors.
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

package latencyslo

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"
	"time"

	k8stypes "k8s.io/apimachinery/pkg/types"

	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrlatency "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/latency"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
)

func makeLatencyAdmissionEndpoint(name string, kvCache float64, runningRequests int) fwksched.Endpoint {
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: name}},
		&fwkdl.Metrics{
			KVCacheUsagePercent: kvCache,
			RunningRequestsSize: runningRequests,
			UpdateTime:          time.Now(),
		},
		nil,
	)
}

func makeSheddableRequest(ttftSLO, tpotSLO string) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		Headers: map[string]string{
			ttftSLOHeaderKey: ttftSLO,
			tpotSLOHeaderKey: tpotSLO,
		},
		Objectives: fwksched.RequestObjectives{Priority: -1},
	}
}

func makeNonSheddableRequest(ttftSLO, tpotSLO string) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{
		Headers: map[string]string{
			ttftSLOHeaderKey: ttftSLO,
			tpotSLOHeaderKey: tpotSLO,
		},
		Objectives: fwksched.RequestObjectives{Priority: 1},
	}
}

func TestAdmit(t *testing.T) {
	plugin := NewLatencyAdmission(LatencyAdmissionDefaultConfig)

	tests := []struct {
		name      string
		request   *fwksched.InferenceRequest
		endpoints []fwksched.Endpoint
		setupFn   func(endpoints []fwksched.Endpoint) // set endpoint attributes
		wantErr   bool
	}{
		{
			name:    "nil request — admit",
			request: nil,
			wantErr: false,
		},
		{
			name:    "non-sheddable request — always admit",
			request: makeNonSheddableRequest("100", "30"),
			endpoints: []fwksched.Endpoint{
				makeLatencyAdmissionEndpoint("pod1", 0.5, 5),
			},
			setupFn: func(endpoints []fwksched.Endpoint) {
				// All invalid predictions
				endpoints[0].Put(attrlatency.LatencyPredictionInfoDataKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -50, -10, 150, 40, 0))
			},
			wantErr: false,
		},
		{
			name:    "no SLO headers — admit",
			request: makeSheddableRequest("", ""),
			endpoints: []fwksched.Endpoint{
				makeLatencyAdmissionEndpoint("pod1", 0.5, 5),
			},
			wantErr: false,
		},
		{
			name:    "sheddable, all invalid, all busy, no cold — reject",
			request: makeSheddableRequest("100", "30"),
			endpoints: []fwksched.Endpoint{
				makeLatencyAdmissionEndpoint("pod1", 0.5, 5),
				makeLatencyAdmissionEndpoint("pod2", 0.4, 3),
			},
			setupFn: func(endpoints []fwksched.Endpoint) {
				endpoints[0].Put(attrlatency.LatencyPredictionInfoDataKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -50, -10, 150, 40, 0))
				endpoints[1].Put(attrlatency.LatencyPredictionInfoDataKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -30, -5, 130, 35, 0))
			},
			wantErr: true,
		},
		{
			name: "sheddable, old SLO aliases — reject",
			request: &fwksched.InferenceRequest{
				Headers: map[string]string{
					metadata.OldTTFTSLOHeaderKey: "100",
					metadata.OldTPOTSLOHeaderKey: "30",
				},
				Objectives: fwksched.RequestObjectives{Priority: -1},
			},
			endpoints: []fwksched.Endpoint{
				makeLatencyAdmissionEndpoint("pod1", 0.5, 5),
			},
			setupFn: func(endpoints []fwksched.Endpoint) {
				endpoints[0].Put(attrlatency.LatencyPredictionInfoDataKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -50, -10, 150, 40, 0))
			},
			wantErr: true,
		},
		{
			name: "new SLO headers take precedence over old aliases",
			request: &fwksched.InferenceRequest{
				Headers: map[string]string{
					metadata.TTFTSLOHeaderKey:    "0",
					metadata.OldTTFTSLOHeaderKey: "100",
					metadata.TPOTSLOHeaderKey:    "0",
					metadata.OldTPOTSLOHeaderKey: "30",
				},
				Objectives: fwksched.RequestObjectives{Priority: -1},
			},
			endpoints: []fwksched.Endpoint{
				makeLatencyAdmissionEndpoint("pod1", 0.5, 5),
			},
			setupFn: func(endpoints []fwksched.Endpoint) {
				endpoints[0].Put(attrlatency.LatencyPredictionInfoDataKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -50, -10, 150, 40, 0))
			},
			wantErr: false,
		},
		{
			name:    "sheddable, all invalid, but one pod idle — admit",
			request: makeSheddableRequest("100", "30"),
			endpoints: []fwksched.Endpoint{
				makeLatencyAdmissionEndpoint("pod1", 0.5, 5),
				makeLatencyAdmissionEndpoint("pod2", 0.4, 0), // idle
			},
			setupFn: func(endpoints []fwksched.Endpoint) {
				endpoints[0].Put(attrlatency.LatencyPredictionInfoDataKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -50, -10, 150, 40, 0))
				endpoints[1].Put(attrlatency.LatencyPredictionInfoDataKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -30, -5, 130, 35, 0))
			},
			wantErr: false,
		},
		{
			name:    "sheddable, all invalid, but cold pod exists — admit",
			request: makeSheddableRequest("100", "30"),
			endpoints: []fwksched.Endpoint{
				makeLatencyAdmissionEndpoint("pod1", 0.5, 5),
				makeLatencyAdmissionEndpoint("pod2", 0.01, 3), // cold
			},
			setupFn: func(endpoints []fwksched.Endpoint) {
				endpoints[0].Put(attrlatency.LatencyPredictionInfoDataKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -50, -10, 150, 40, 0))
				endpoints[1].Put(attrlatency.LatencyPredictionInfoDataKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -30, -5, 130, 35, 0))
			},
			wantErr: false,
		},
		{
			name:    "sheddable, one valid endpoint — admit",
			request: makeSheddableRequest("100", "30"),
			endpoints: []fwksched.Endpoint{
				makeLatencyAdmissionEndpoint("pod1", 0.5, 5),
				makeLatencyAdmissionEndpoint("pod2", 0.4, 3),
			},
			setupFn: func(endpoints []fwksched.Endpoint) {
				endpoints[0].Put(attrlatency.LatencyPredictionInfoDataKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -50, -10, 150, 40, 0))
				endpoints[1].Put(attrlatency.LatencyPredictionInfoDataKey,
					attrlatency.NewLatencyPredictionInfo(true, true, 20, 5, 80, 25, 0)) // valid
			},
			wantErr: false,
		},
		{
			name:    "sheddable, no prediction data on endpoints — admit (fail-open)",
			request: makeSheddableRequest("100", "30"),
			endpoints: []fwksched.Endpoint{
				makeLatencyAdmissionEndpoint("pod1", 0.5, 5),
			},
			// no setupFn — no latency attributes set
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.setupFn != nil {
				tt.setupFn(tt.endpoints)
			}
			err := plugin.Admit(context.Background(), tt.request, tt.endpoints)
			if (err != nil) != tt.wantErr {
				t.Errorf("Admit() error = %v, wantErr %v", err, tt.wantErr)
			}
			// A rejection is a shed under load: 429 with the saturated dropped reason, not a 500.
			var e errcommon.Error
			if err != nil && (!errors.As(err, &e) || e.Code != errcommon.ResourceExhausted ||
				e.Headers[errcommon.RequestDroppedReasonHeaderKey] != string(errcommon.RequestDroppedReasonSaturated)) {
				t.Errorf("Admit() rejection = %#v, want ResourceExhausted with dropped reason %q", err, errcommon.RequestDroppedReasonSaturated)
			}
		})
	}
}

func TestAdmitNonFiniteSLOHeaders(t *testing.T) {
	plugin := NewLatencyAdmission(LatencyAdmissionDefaultConfig)

	for _, value := range []string{"NaN", "+Inf", "-Inf", "Infinity", "inf"} {
		for _, tt := range []struct {
			name    string
			request *fwksched.InferenceRequest
			wantErr bool
		}{
			{
				name:    "TTFT only",
				request: makeSheddableRequest(value, ""),
			},
			{
				name:    "TPOT only",
				request: makeSheddableRequest("", value),
			},
			{
				name:    "valid TPOT",
				request: makeSheddableRequest(value, "30"),
				wantErr: true,
			},
			{
				name:    "valid TTFT",
				request: makeSheddableRequest("100", value),
				wantErr: true,
			},
		} {
			t.Run(value+"/"+tt.name, func(t *testing.T) {
				endpoint := makeLatencyAdmissionEndpoint("pod1", 0.5, 5)
				endpoint.Put(attrlatency.LatencyPredictionInfoDataKey,
					attrlatency.NewLatencyPredictionInfo(false, false, -150, -40, 150, 40, 0))

				err := plugin.Admit(context.Background(), tt.request, []fwksched.Endpoint{endpoint})
				if (err != nil) != tt.wantErr {
					t.Errorf("Admit() error = %v, wantErr %v", err, tt.wantErr)
				}
			})
		}
	}
}

// A never-scraped endpoint carries a zero-value metrics snapshot. Its zero KV
// usage, zero running requests and the prediction made from them are not
// observations, so they must not let a sheddable request through.
func TestAdmitIgnoresNeverScrapedEndpoints(t *testing.T) {
	raw, err := LatencyAdmissionFactory("latency-admitter", json.NewDecoder(strings.NewReader(`{}`)), nil)
	if err != nil {
		t.Fatalf("LatencyAdmissionFactory() error = %v", err)
	}
	plugin, ok := raw.(*LatencyAdmission)
	if !ok {
		t.Fatalf("LatencyAdmissionFactory() returned %T, want *LatencyAdmission", raw)
	}

	busy := makeLatencyAdmissionEndpoint("busy", 0.5, 5)
	busy.Put(attrlatency.LatencyPredictionInfoDataKey,
		attrlatency.NewLatencyPredictionInfo(false, false, -50, -10, 150, 40, 0))
	// The predictor sees the same zero-value metrics, so its prediction for a
	// never-scraped endpoint looks valid.
	neverScraped := fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{ID: k8stypes.NamespacedName{Name: "never-scraped"}},
		fwkdl.NewMetrics(),
		nil,
	)
	neverScraped.Put(attrlatency.LatencyPredictionInfoDataKey,
		attrlatency.NewLatencyPredictionInfo(true, true, 50, 10, 50, 20, 0))

	err = plugin.Admit(context.Background(), makeSheddableRequest("100", "30"),
		[]fwksched.Endpoint{busy, neverScraped})
	var e errcommon.Error
	if !errors.As(err, &e) || e.Code != errcommon.ResourceExhausted {
		t.Fatalf("Admit() error = %v, want a ResourceExhausted shed: the never-scraped endpoint is no evidence", err)
	}

	// With no scraped endpoint at all there is nothing to judge load by, so admit.
	if err := plugin.Admit(context.Background(), makeSheddableRequest("100", "30"),
		[]fwksched.Endpoint{neverScraped}); err != nil {
		t.Errorf("Admit() with only never-scraped endpoints error = %v, want admit", err)
	}
}
