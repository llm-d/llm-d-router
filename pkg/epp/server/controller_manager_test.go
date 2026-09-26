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

package server

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	apixv1 "github.com/llm-d/llm-d-router/apix/v1"
	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	"github.com/llm-d/llm-d-router/pkg/common"
)

func TestObjectiveSchemeAndCacheFollowPrimary(t *testing.T) {
	gknn := common.GKNN{}
	tests := []struct {
		name      string
		cfg       ControllerConfig
		wantV1    bool
		wantAlpha bool
	}{
		{
			name: "v1 primary registers and caches v1 only",
			cfg: ControllerConfig{
				startCrdReconcilers:     true,
				hasInferenceObjective:   true,
				InferenceObjectiveGV:    inferenceObjectiveV1GV,
				hasV1InferenceObjective: true,
			},
			wantV1: true,
		},
		{
			name: "v1alpha2 primary registers and caches v1alpha2 only",
			cfg: ControllerConfig{
				startCrdReconcilers:   true,
				hasInferenceObjective: true,
				InferenceObjectiveGV:  inferenceAPIGV,
			},
			wantAlpha: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			scheme := NewScheme(tt.cfg)
			v1GVK := schema.GroupVersionKind{Group: "llm-d.ai", Version: "v1", Kind: "InferenceObjective"}
			alphaGVK := schema.GroupVersionKind{Group: "llm-d.ai", Version: "v1alpha2", Kind: "InferenceObjective"}
			if got := scheme.Recognizes(v1GVK); got != tt.wantV1 {
				t.Errorf("scheme recognizes llm-d.ai/v1 = %v, want %v", got, tt.wantV1)
			}
			if got := scheme.Recognizes(alphaGVK); got != tt.wantAlpha {
				t.Errorf("scheme recognizes llm-d.ai/v1alpha2 = %v, want %v", got, tt.wantAlpha)
			}
			opt := defaultManagerOptions(tt.cfg, gknn, metricsserver.Options{}, scheme)
			var hasV1, hasAlpha bool
			for obj := range opt.Cache.ByObject {
				switch obj.(type) {
				case *apixv1.InferenceObjective:
					hasV1 = true
				case *v1alpha2.InferenceObjective:
					hasAlpha = true
				}
			}
			if hasV1 != tt.wantV1 {
				t.Errorf("cache watches llm-d.ai/v1 = %v, want %v", hasV1, tt.wantV1)
			}
			if hasAlpha != tt.wantAlpha {
				t.Errorf("cache watches llm-d.ai/v1alpha2 = %v, want %v", hasAlpha, tt.wantAlpha)
			}
		})
	}
}
