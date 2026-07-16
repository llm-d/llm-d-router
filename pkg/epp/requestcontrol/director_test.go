/*
Copyright 2025 The Kubernetes Authors.

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

package requestcontrol

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrprefix "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/prefix"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/parsers/openai"
	sessionaffinityfilter "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/filter/sessionaffinity"
	sessionaffinityscorer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/scorer/sessionaffinity"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
	"github.com/llm-d/llm-d-router/pkg/epp/metadata"
	poolutil "github.com/llm-d/llm-d-router/pkg/epp/util/pool"
	testutil "github.com/llm-d/llm-d-router/pkg/epp/util/testing"
)

var (
	mockProducedDataKey = fwkplugin.NewDataKey("producedDataKey", "mock-producer")
)

// --- Mocks ---

type mockAdmissionController struct {
	admitErr error
}

func (m *mockAdmissionController) Admit(context.Context, *handlers.RequestContext, int) error {
	return m.admitErr
}

// mockConcurrencyReleaserAdmissionController additionally implements
// ConcurrencyReleaser, mirroring FlowControlAdmissionController when
// WithConcurrencyLease is configured (RFC #1593 feasibility spike).
type mockConcurrencyReleaserAdmissionController struct {
	mockAdmissionController
	releaseCalls  int
	lastRequestID string
	lastFairness  string
	lastPriority  int
}

func (m *mockConcurrencyReleaserAdmissionController) ReleaseConcurrency(_ context.Context, requestID, fairnessID string, priority int) {
	m.releaseCalls++
	m.lastRequestID = requestID
	m.lastFairness = fairnessID
	m.lastPriority = priority
}

var _ ConcurrencyReleaser = (*mockConcurrencyReleaserAdmissionController)(nil)

type mockScheduler struct {
	scheduleResults *fwksched.SchedulingResult
	scheduleErr     error
	dataProduced    bool // denotes whether data production is expected.
}

func (m *mockScheduler) Schedule(_ context.Context, _ *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) (*fwksched.SchedulingResult, error) {
	if endpoints != nil && m.dataProduced {
		data, ok := endpoints[0].Get(mockProducedDataKey.String())
		if !ok || data.(mockProducedDataType).value != 42 {
			return nil, errors.New("expected produced data not found in pod")
		}
	}
	return m.scheduleResults, m.scheduleErr
}

type mockDatastore struct {
	pods     []fwkdl.Endpoint
	rewrites []*v1alpha2.InferenceModelRewrite
}

func (ds *mockDatastore) PoolGet() (*datalayer.EndpointPool, error) {
	return nil, errors.New("sentinel error for mock datastore")
}
func (ds *mockDatastore) ObjectiveGet(_ string) *v1alpha2.InferenceObjective {
	return nil
}
func (ds *mockDatastore) PodList(predicate func(fwkdl.Endpoint) bool) []fwkdl.Endpoint {
	res := []fwkdl.Endpoint{}
	for _, pod := range ds.pods {
		if predicate(pod) {
			res = append(res, pod)
		}
	}

	return res
}

type mockDataProducerPlugin struct {
	name     string
	produces map[fwkplugin.DataKey]any
	consumes map[fwkplugin.DataKey]any
}

func (m *mockDataProducerPlugin) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Name: m.name, Type: "mock"}
}

func (m *mockDataProducerPlugin) Produces() map[fwkplugin.DataKey]any {
	return m.produces
}

func (m *mockDataProducerPlugin) Consumes() fwkplugin.DataDependencies {
	return fwkplugin.DataDependencies{Required: m.consumes}
}

func (m *mockDataProducerPlugin) Produce(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	endpoints[0].Put(mockProducedDataKey.String(), mockProducedDataType{value: 42})
	return nil
}

func newMockDataProducerPlugin(name string) *mockDataProducerPlugin {
	return &mockDataProducerPlugin{
		name:     name,
		produces: map[fwkplugin.DataKey]any{mockProducedDataKey: 0},
		consumes: map[fwkplugin.DataKey]any{},
	}
}

type mockAdmissionPlugin struct {
	typedName   fwkplugin.TypedName
	denialError error
}

func newMockAdmissionPlugin(name string, denialError error) *mockAdmissionPlugin {
	return &mockAdmissionPlugin{
		typedName:   fwkplugin.TypedName{Type: "mock-admit-data", Name: name},
		denialError: denialError,
	}
}

func (m *mockAdmissionPlugin) TypedName() fwkplugin.TypedName {
	return m.typedName
}

func (m *mockAdmissionPlugin) Admit(ctx context.Context, request *fwksched.InferenceRequest, endpoints []fwksched.Endpoint) error {
	return m.denialError
}

type mockPreRequestPlugin struct {
	name     string
	modifyFn func(request *fwksched.InferenceRequest)
}

func (m *mockPreRequestPlugin) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Name: m.name, Type: "mock"}
}

func (m *mockPreRequestPlugin) PreRequest(ctx context.Context, request *fwksched.InferenceRequest, schedulingResult *fwksched.SchedulingResult) {
	if m.modifyFn != nil {
		m.modifyFn(request)
	}
}

type mockProducedDataType struct {
	value int
}

// Clone implements types.Cloneable.
func (m mockProducedDataType) Clone() fwkdl.Cloneable {
	return mockProducedDataType{value: m.value}
}

func (ds *mockDatastore) ModelRewriteGet(modelName string) (*v1alpha2.InferenceModelRewriteRule, string) {
	// This mock implementation simulates the precedence logic for simplicity.
	// It finds the oldest rewrite that has a rule matching the modelName,
	// prioritizing exact matches over generic (empty Matches) rules.
	var matchingRewrites []*v1alpha2.InferenceModelRewrite
	var genericRewrites []*v1alpha2.InferenceModelRewrite
	for _, r := range ds.rewrites {
		for _, rule := range r.Spec.Rules {
			if len(rule.Matches) == 0 {
				genericRewrites = append(genericRewrites, r)
				continue
			}
			for _, match := range rule.Matches {
				if match.Model != nil && match.Model.Value == modelName {
					matchingRewrites = append(matchingRewrites, r)
					break // break inner loop
				}
			}
		}
	}

	if len(matchingRewrites) == 0 && len(genericRewrites) == 0 {
		return nil, ""
	}

	// Exact matches take precedence; fall back to generic rules.
	candidates := matchingRewrites
	if len(candidates) == 0 {
		candidates = genericRewrites
	}

	// Sort by timestamp to find the oldest.
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].CreationTimestamp.Before(&candidates[j].CreationTimestamp)
	})

	// Return the first rule from the oldest rewrite.
	return &candidates[0].Spec.Rules[0], candidates[0].Name
}

func TestDirector_HandleRequest(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	// --- Setup common objects ---
	model := "food-review"
	modelSheddable := "food-review-sheddable"
	modelWithResolvedTarget := "food-review-resolve"
	modelToBeRewritten := "food-review-to-be-rewritten"
	modelRewritten := "food-review-rewritten"

	objectiveName := "ioFoodReview"
	objectiveNameSheddable := "imFoodReviewSheddable"
	objectiveNameResolve := "imFoodReviewResolve"
	// InferenceObjective definitions
	ioFoodReview := testutil.MakeInferenceObjective("ioFoodReview").
		CreationTimestamp(metav1.Unix(1000, 0)).
		Priority(2).
		ObjRef()
	ioFoodReviewSheddable := testutil.MakeInferenceObjective("imFoodReviewSheddable").
		CreationTimestamp(metav1.Unix(1000, 0)).
		Priority(-1).
		ObjRef()
	ioFoodReviewResolve := testutil.MakeInferenceObjective("imFoodReviewResolve").
		CreationTimestamp(metav1.Unix(1000, 0)).
		Priority(1).
		ObjRef()

	rewrite := &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "rewrite-rule",
			CreationTimestamp: metav1.Now(),
		},
		Spec: v1alpha2.InferenceModelRewriteSpec{
			Rules: []v1alpha2.InferenceModelRewriteRule{
				{
					Matches: []v1alpha2.Match{
						{
							Model: &v1alpha2.ModelMatch{
								Value: modelToBeRewritten,
							},
						},
					},
					Targets: []v1alpha2.TargetModel{
						{
							ModelRewrite: modelRewritten,
							Weight:       ptr.To[int32](100),
						},
					},
				},
			},
		},
	}

	genericRewriteTarget := "generic-target-model"
	genericRewrite := &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "generic-rewrite-rule",
			CreationTimestamp: metav1.Now(),
		},
		Spec: v1alpha2.InferenceModelRewriteSpec{
			Rules: []v1alpha2.InferenceModelRewriteRule{
				{
					Targets: []v1alpha2.TargetModel{
						{
							ModelRewrite: genericRewriteTarget,
							Weight:       ptr.To[int32](100),
						},
					},
				},
			},
		},
	}

	pool := &v1.InferencePool{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pool", Namespace: "default"},
		Spec: v1.InferencePoolSpec{
			TargetPorts: []v1.Port{{Number: v1.PortNumber(int32(8000))}},
			Selector: v1.LabelSelector{
				MatchLabels: map[v1.LabelKey]v1.LabelValue{
					"app": "inference",
				},
			},
		},
	}

	defaultSuccessfulScheduleResults := &fwksched.SchedulingResult{
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			"testProfile": {
				TargetEndpoints: []fwksched.Endpoint{
					&fwksched.ScoredEndpoint{
						Endpoint: fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
							Address:        "192.168.1.100",
							Port:           "8000",
							MetricsHost:    "192.168.1.100:8000",
							NamespacedName: types.NamespacedName{Name: "pod1", Namespace: "default"},
						}, nil, nil),
					},
					&fwksched.ScoredEndpoint{
						Endpoint: fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
							Address:        "192.168.2.100",
							Port:           "8000",
							MetricsHost:    "192.168.2.100:8000",
							NamespacedName: types.NamespacedName{Name: "pod2", Namespace: "default"},
						}, nil, nil),
					},
					&fwksched.ScoredEndpoint{
						Endpoint: fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
							Address:        "192.168.4.100",
							Port:           "8000",
							MetricsHost:    "192.168.4.100:8000",
							NamespacedName: types.NamespacedName{Name: "pod4", Namespace: "default"},
						}, nil, nil),
					},
				},
			},
		},
		PrimaryProfileName: "testProfile",
	}

	tests := []struct {
		name                    string
		reqBodyMap              map[string]any
		mockAdmissionController *mockAdmissionController
		inferenceObjectiveName  string
		schedulerMockSetup      func(m *mockScheduler)
		initialTargetModelName  string // Initial target model in the reqCtx.
		parser                  fwkrh.Parser
		wantErrCode             string                   // Expected errcommon code string
		wantReqCtx              *handlers.RequestContext // Fields to check in the returned RequestContext
		targetModelName         string                   // Expected model name after target model resolution
		admitRequestDenialError error                    // Expected denial error from admission plugin
		dataProducerPlugin      *mockDataProducerPlugin
		preRequestPlugin        *mockPreRequestPlugin
		wantMutatedBody         map[string]any
		fairnessIDHeader        string // If non-empty, set as metadata.FlowFairnessIDKey on the incoming request.
		wantFairnessID          string // If non-empty, asserted against returnedReqCtx.SchedulingRequest.FairnessID.
		rewrites                []*v1alpha2.InferenceModelRewrite
	}{
		{
			name: "successful completions request",
			reqBodyMap: map[string]any{
				"model":  model,
				"prompt": "critical prompt",
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			initialTargetModelName: model,
			wantReqCtx: &handlers.RequestContext{
				ObjectiveKey:    objectiveName,
				TargetModelName: model,
				TargetPod: &fwkdl.EndpointMetadata{
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
					Port:           "8000",
					MetricsHost:    "192.168.1.100:8000",
				},
				TargetEndpoint: "192.168.1.100:8000,192.168.2.100:8000,192.168.4.100:8000",
			},
			wantMutatedBody: map[string]any{
				"model":  model,
				"prompt": "critical prompt",
			},
			inferenceObjectiveName: objectiveName,
		},
		{
			name: "fairness ID derived from header",
			reqBodyMap: map[string]any{
				"model":  model,
				"prompt": "critical prompt",
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			initialTargetModelName: model,
			inferenceObjectiveName: objectiveName,
			fairnessIDHeader:       "user-123",
			wantFairnessID:         "user-123",
		},
		{
			name: "fairness ID falls back to default when header absent",
			reqBodyMap: map[string]any{
				"model":  model,
				"prompt": "critical prompt",
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			initialTargetModelName: model,
			inferenceObjectiveName: objectiveName,
			wantFairnessID:         metadata.DefaultFairnessID,
		},
		{
			name: "successful request with preRequest plugin adding key",
			reqBodyMap: map[string]any{
				"model":  model,
				"prompt": "original prompt",
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			initialTargetModelName: model,
			wantReqCtx: &handlers.RequestContext{
				ObjectiveKey:    objectiveName,
				TargetModelName: model,
				TargetPod: &fwkdl.EndpointMetadata{
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
					Port:           "8000",
					MetricsHost:    "192.168.1.100:8000",
				},
				TargetEndpoint: "192.168.1.100:8000,192.168.2.100:8000,192.168.4.100:8000",
			},
			wantMutatedBody: map[string]any{
				"model":   model,
				"prompt":  "original prompt",
				"new_key": "new_value",
			},
			inferenceObjectiveName: objectiveName,
			preRequestPlugin: &mockPreRequestPlugin{
				name: "test-pre-request-plugin",
				modifyFn: func(request *fwksched.InferenceRequest) {
					if payloadMap, ok := request.Body.Payload.(fwkrh.PayloadMap); ok {
						payloadMap["new_key"] = "new_value"
					}
				},
			},
		}, {
			name: "successful request with model rewrite",
			reqBodyMap: map[string]any{
				"model":  modelToBeRewritten,
				"prompt": "some prompt",
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			initialTargetModelName: model,
			wantReqCtx: &handlers.RequestContext{
				ObjectiveKey:    model,
				TargetModelName: modelRewritten,
				TargetPod: &fwkdl.EndpointMetadata{
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
					Port:           "8000",
					MetricsHost:    "192.168.1.100:8000",
				},
				TargetEndpoint: "192.168.1.100:8000,192.168.2.100:8000,192.168.4.100:8000",
			},
			wantMutatedBody: map[string]any{
				"model":  modelRewritten,
				"prompt": "some prompt",
			},
			inferenceObjectiveName: model,
			rewrites:               []*v1alpha2.InferenceModelRewrite{rewrite},
		}, {
			name: "successful chat completions request",
			reqBodyMap: map[string]any{
				"model": model,
				"messages": []any{
					map[string]any{
						"role":    "user",
						"content": "critical prompt",
					},
				},
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			initialTargetModelName: model,
			wantReqCtx: &handlers.RequestContext{
				TargetModelName: model,
				TargetPod: &fwkdl.EndpointMetadata{
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
					Port:           "8000",
					MetricsHost:    "192.168.1.100:8000",
				},
				TargetEndpoint: "192.168.1.100:8000,192.168.2.100:8000,192.168.4.100:8000",
			},
			wantMutatedBody: map[string]any{
				"model": model,
				"messages": []any{
					map[string]any{
						"role":    "user",
						"content": "critical prompt",
					},
				},
			},
			targetModelName: model,
		},
		{
			name: "successful chat completions request with DataProducer plugins",
			reqBodyMap: map[string]any{
				"model": model,
				"messages": []any{
					map[string]any{
						"role":    "user",
						"content": "critical prompt",
					},
				},
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
				m.dataProduced = true
			},
			wantReqCtx: &handlers.RequestContext{
				TargetModelName: model,
				TargetPod: &fwkdl.EndpointMetadata{
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
					Port:           "8000",
					MetricsHost:    "192.168.1.100:8000",
				},
				TargetEndpoint: "192.168.1.100:8000,192.168.2.100:8000,192.168.4.100:8000",
			},
			wantMutatedBody: map[string]any{
				"model": model,
				"messages": []any{
					map[string]any{
						"role":    "user",
						"content": "critical prompt",
					},
				},
			},
			targetModelName:    model,
			dataProducerPlugin: newMockDataProducerPlugin("test-plugin"),
		},
		{
			name: "successful chat completions request with admit request plugins",
			reqBodyMap: map[string]any{
				"model": model,
				"messages": []any{
					map[string]any{
						"role":    "user",
						"content": "critical prompt",
					},
				},
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			wantReqCtx: &handlers.RequestContext{
				TargetModelName: model,
				TargetPod: &fwkdl.EndpointMetadata{
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
					Port:           "8000",
					MetricsHost:    "192.168.1.100:8000",
				},
				TargetEndpoint: "192.168.1.100:8000,192.168.2.100:8000,192.168.4.100:8000",
			},
			wantMutatedBody: map[string]any{
				"model": model,
				"messages": []any{
					map[string]any{
						"role":    "user",
						"content": "critical prompt",
					},
				},
			},
			targetModelName:         model,
			admitRequestDenialError: nil,
		},
		{
			name: "denied request by admit request plugin",
			reqBodyMap: map[string]any{
				"model": model,
				"messages": []any{
					map[string]any{
						"role":    "user",
						"content": "critical prompt",
					},
				},
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			wantMutatedBody: map[string]any{
				"model": model,
				"messages": []any{
					map[string]any{
						"role":    "user",
						"content": "critical prompt",
					},
				},
			},
			targetModelName:         model,
			admitRequestDenialError: errors.New("denied by admit plugin"),
			wantErrCode:             errcommon.Internal,
		},
		{
			name: "successful chat completions request with multiple messages",
			reqBodyMap: map[string]any{
				"model": model,
				"messages": []any{
					map[string]any{
						"role":    "developer",
						"content": "You are a helpful assistant.",
					},
					map[string]any{
						"role":    "user",
						"content": "Hello!",
					},
				},
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			initialTargetModelName: model,
			wantReqCtx: &handlers.RequestContext{
				ObjectiveKey:    objectiveName,
				TargetModelName: model,
				TargetPod: &fwkdl.EndpointMetadata{
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
					Port:           "8000",
					MetricsHost:    "192.168.1.100:8000",
				},
				TargetEndpoint: "192.168.1.100:8000,192.168.2.100:8000,192.168.4.100:8000",
			},
			inferenceObjectiveName: objectiveName,
		}, {
			name: "successful request with target model resolution",
			reqBodyMap: map[string]any{
				"model":  modelWithResolvedTarget,
				"prompt": "prompt for target resolution",
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			initialTargetModelName: "resolved-target-model-A",
			wantReqCtx: &handlers.RequestContext{
				ObjectiveKey:    objectiveNameResolve,
				TargetModelName: "resolved-target-model-A",
				TargetPod: &fwkdl.EndpointMetadata{
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
					Port:           "8000",
					MetricsHost:    "192.168.1.100:8000",
				},
				TargetEndpoint: "192.168.1.100:8000,192.168.2.100:8000,192.168.4.100:8000",
			},
			wantMutatedBody: map[string]any{
				"model":  "resolved-target-model-A",
				"prompt": "prompt for target resolution",
			},
			inferenceObjectiveName: objectiveNameResolve,
		},
		{
			name: "nonexistent target defined, use default inference model",
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			initialTargetModelName: "food-review-1",
			wantReqCtx: &handlers.RequestContext{
				ObjectiveKey:    "food-review-1",
				TargetModelName: "food-review-1",
				TargetPod: &fwkdl.EndpointMetadata{
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
					Port:           "8000",
					MetricsHost:    "192.168.1.100:8000",
				},
				TargetEndpoint: "192.168.1.100:8000,192.168.2.100:8000,192.168.4.100:8000",
			},
			wantMutatedBody: map[string]any{
				"model":  "food-review-1",
				"prompt": "test prompt",
			},
			reqBodyMap: map[string]any{
				"model":  "food-review-1",
				"prompt": "test prompt",
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			inferenceObjectiveName:  "food-review-1",
		},
		{
			name: "request rejected by admission controller",
			reqBodyMap: map[string]any{
				"model":  modelSheddable,
				"prompt": "sheddable prompt",
			},
			inferenceObjectiveName:  objectiveNameSheddable,
			mockAdmissionController: &mockAdmissionController{admitErr: errcommon.Error{Code: errcommon.ResourceExhausted, Msg: "simulated admission rejection"}},
			wantErrCode:             errcommon.ResourceExhausted,
		},
		{
			name:                    "model not found, expect err",
			reqBodyMap:              map[string]any{"prompt": "p"},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			wantErrCode:             errcommon.BadRequest,
		},
		{
			name:                    "missing model field resolved by generic rewrite",
			reqBodyMap:              map[string]any{"prompt": "p"},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = defaultSuccessfulScheduleResults
			},
			wantReqCtx: &handlers.RequestContext{
				TargetModelName: genericRewriteTarget,
				TargetPod: &fwkdl.EndpointMetadata{
					NamespacedName: types.NamespacedName{Namespace: "default", Name: "pod1"},
					Address:        "192.168.1.100",
					Port:           "8000",
					MetricsHost:    "192.168.1.100:8000",
				},
				TargetEndpoint: "192.168.1.100:8000,192.168.2.100:8000,192.168.4.100:8000",
			},
			wantMutatedBody: map[string]any{
				"model":  genericRewriteTarget,
				"prompt": "p",
			},
			rewrites: []*v1alpha2.InferenceModelRewrite{genericRewrite},
		},
		{
			name:        "prompt or messages not found, expect err",
			reqBodyMap:  map[string]any{"model": model},
			wantErrCode: errcommon.BadRequest,
		},
		{
			name: "empty messages, expect err",
			reqBodyMap: map[string]any{
				"model":    model,
				"messages": []any{},
			},
			wantErrCode: errcommon.BadRequest,
		},
		{
			name: "scheduler returns error",
			reqBodyMap: map[string]any{
				"model":  model,
				"prompt": "prompt that causes scheduler error",
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleErr = errors.New("simulated scheduler failure")
			},
			wantErrCode:            errcommon.ResourceExhausted,
			inferenceObjectiveName: objectiveName,
		},
		{
			name: "scheduler returns nil result and nil error",
			reqBodyMap: map[string]any{
				"model":  model,
				"prompt": "prompt for nil,nil scheduler return",
			},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) {
				m.scheduleResults = nil
				m.scheduleErr = nil
			},
			wantErrCode:            errcommon.Internal,
			inferenceObjectiveName: objectiveName,
		},
	}

	period := time.Second
	factories := []datalayer.EndpointFactory{
		datalayer.NewTestRuntime(t, period),
	}
	for _, epf := range factories {
		// Datastore setup
		ds := datastore.NewDatastore(t.Context(), epf)
		ds.ObjectiveSet(ioFoodReview)
		ds.ObjectiveSet(ioFoodReviewResolve)
		ds.ObjectiveSet(ioFoodReviewSheddable)
		ds.ModelRewriteSet(rewrite)

		scheme := runtime.NewScheme()
		_ = clientgoscheme.AddToScheme(scheme)
		fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()

		if err := ds.PoolSet(ctx, fakeClient, poolutil.InferencePoolToEndpointPool(pool)); err != nil {
			t.Fatalf("Error while setting inference pool: %v", err)
		}

		for i := range 5 {
			// Pod setup
			testPod := &corev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      fmt.Sprintf("pod%v", i+1),
					Namespace: "default",
					Labels:    map[string]string{"app": "inference"},
				},
				Status: corev1.PodStatus{
					PodIP:      fmt.Sprintf("192.168.%v.100", i+1),
					Phase:      corev1.PodRunning,
					Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}},
				},
			}
			ds.PodUpdateOrAddIfNotExist(ctx, testPod)
		}

		for _, test := range tests {
			t.Run(test.name, func(t *testing.T) {
				mockSched := &mockScheduler{}
				if test.schedulerMockSetup != nil {
					test.schedulerMockSetup(mockSched)
				}
				config := NewConfig()
				if test.dataProducerPlugin != nil {
					config = config.WithDataProducerPlugins(test.dataProducerPlugin)
				}
				if test.preRequestPlugin != nil {
					config = config.WithPreRequestPlugins(test.preRequestPlugin)
				}
				config = config.WithAdmissionPlugins(newMockAdmissionPlugin("test-admit-plugin", test.admitRequestDenialError))

				endpointCandidates := NewCachedEndpointCandidates(context.Background(), NewDatastoreEndpointCandidates(ds), time.Minute)
				director := NewDirectorWithConfig(ds, mockSched, test.mockAdmissionController, endpointCandidates, config)
				if len(test.rewrites) > 0 {
					mockDs := &mockDatastore{
						pods:     ds.PodList(datastore.AllPodsPredicate),
						rewrites: test.rewrites,
					}
					director.datastore = mockDs
					director.endpointCandidates = NewCachedEndpointCandidates(context.Background(), NewDatastoreEndpointCandidates(mockDs), time.Minute)
				}

				reqCtx := &handlers.RequestContext{
					Request: &handlers.Request{
						Headers: map[string]string{
							reqcommon.RequestIDHeaderKey: "test-req-id-" + test.name, // Ensure a default request ID
						},
					},
					ObjectiveKey:    test.inferenceObjectiveName,
					TargetModelName: test.initialTargetModelName,
				}
				var err error
				reqCtx.Request.RawBody, err = json.Marshal(test.reqBodyMap)
				if err != nil {
					t.Fatalf("Error parsing the reqBodyMap, err is %v", err)
				}

				// Add appropriate path header based on request body content for path-based API detection
				if _, hasPrompt := test.reqBodyMap["prompt"]; hasPrompt {
					reqCtx.Request.Headers[":path"] = "/v1/completions"
				} else if _, hasMessages := test.reqBodyMap["messages"]; hasMessages {
					reqCtx.Request.Headers[":path"] = "/v1/chat/completions"
				}

				if test.fairnessIDHeader != "" {
					reqCtx.Request.Headers[metadata.FlowFairnessIDKey] = test.fairnessIDHeader
				}

				parseResult, parseErr := openai.NewOpenAIParser().ParseRequest(ctx, reqCtx.Request.RawBody, reqCtx.Request.Headers)
				var returnedReqCtx *handlers.RequestContext
				if parseErr != nil {
					err = errcommon.Error{Code: errcommon.BadRequest, Msg: parseErr.Error()}
				} else {
					returnedReqCtx, err = director.HandleRequest(ctx, reqCtx, parseResult.Body)
				}

				if test.wantErrCode != "" {
					assert.Error(t, err, "HandleRequest() should have returned an error")
					var e errcommon.Error
					if assert.ErrorAs(t, err, &e, "Error should be of type errcommon.Error") {
						assert.Equal(t, test.wantErrCode, e.Code, "Error code mismatch")
					}
					return
				}

				assert.NoError(t, err, "HandleRequest() returned unexpected error")

				if test.wantReqCtx != nil {
					assert.Equal(t, test.wantReqCtx.ObjectiveKey, returnedReqCtx.ObjectiveKey, "reqCtx.Model mismatch")
					assert.Equal(t, test.wantReqCtx.TargetModelName, returnedReqCtx.TargetModelName,
						"reqCtx.ResolvedTargetModel mismatch")
					if diff := cmp.Diff(test.wantReqCtx.TargetPod, returnedReqCtx.TargetPod, cmpopts.EquateEmpty()); diff != "" {
						t.Errorf("reqCtx.TargetPod mismatch (-want +got):\n%s", diff)
					}
					assert.Equal(t, test.wantReqCtx.TargetEndpoint, returnedReqCtx.TargetEndpoint, "reqCtx.TargetEndpoint mismatch")
				}

				if test.wantFairnessID != "" {
					require.NotNil(t, returnedReqCtx.SchedulingRequest, "SchedulingRequest should be populated")
					assert.Equal(t, test.wantFairnessID, returnedReqCtx.SchedulingRequest.FairnessID, "SchedulingRequest.FairnessID mismatch")
				}

				if test.wantMutatedBody != nil {
					assert.NotEmpty(t, returnedReqCtx.Request.RawBody, "Expected mutated body, but reqCtx.Request.Body is nil")
					updatedBodyMap := make(map[string]any)
					if err := json.Unmarshal(reqCtx.Request.RawBody, &updatedBodyMap); err != nil {
						t.Errorf("Error to Unmarshal reqCtx.Request.UpdatedBody, err is %v", err)
					}
					if diff := cmp.Diff(test.wantMutatedBody, updatedBodyMap); diff != "" {
						t.Errorf("reqCtx.Request.RawBody mismatch (-want +got):\n%s", diff)
					}
				}
				assert.Equal(t, len(reqCtx.Request.RawBody), reqCtx.RequestSize)
			})
		}
	}
}

func TestGetRandomEndpoint(t *testing.T) {
	tests := []struct {
		name      string
		storePods []*corev1.Pod
		expectNil bool
	}{
		{
			name:      "No pods available",
			storePods: []*corev1.Pod{},
			expectNil: true,
		},
		{
			name: "Single pod available",
			storePods: []*corev1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}},
			},
			expectNil: false,
		},
		{
			name: "Multiple pods available",
			storePods: []*corev1.Pod{
				{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "pod2"}},
				{ObjectMeta: metav1.ObjectMeta{Name: "pod3"}},
			},
			expectNil: false,
		},
	}

	scheme := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(scheme)
	_ = v1alpha2.Install(scheme)
	_ = v1.Install(scheme)
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		Build()
	pool := &v1.InferencePool{
		Spec: v1.InferencePoolSpec{
			TargetPorts: []v1.Port{
				{Number: 8000},
			},
		},
	}

	for _, test := range tests {
		period := time.Millisecond
		factories := []datalayer.EndpointFactory{
			datalayer.NewTestRuntime(t, period),
		}
		for _, epf := range factories {
			t.Run(test.name, func(t *testing.T) {
				endpointPool := poolutil.InferencePoolToEndpointPool(pool)
				ds := datastore.NewDatastore(t.Context(), epf)
				err := ds.PoolSet(t.Context(), fakeClient, endpointPool)
				if err != nil {
					t.Errorf("unexpected error setting pool: %s", err)
				}
				for _, pod := range test.storePods {
					ds.PodUpdateOrAddIfNotExist(context.Background(), pod)
				}
				d := &Director{datastore: ds}
				gotEndpoint := d.GetRandomEndpoint()

				if test.expectNil && gotEndpoint != nil {
					t.Errorf("expected nil pod, got: %v", gotEndpoint)
				}
				if !test.expectNil && gotEndpoint == nil {
					t.Errorf("expected non-nil pod, got nil")
				}
			})
		}
	}
}

func TestDirector_ApplyWeightedModelRewrite(t *testing.T) {
	_ = logutil.NewTestLoggerIntoContext(context.Background())

	// Mock InferenceModelRewrite objects
	rewriteOld := &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "rewrite-old",
			CreationTimestamp: metav1.Unix(1000, 0),
		},
		Spec: v1alpha2.InferenceModelRewriteSpec{
			Rules: []v1alpha2.InferenceModelRewriteRule{
				{
					Matches: []v1alpha2.Match{
						{
							Model: &v1alpha2.ModelMatch{
								Value: "model-a",
							},
						},
					},
					Targets: []v1alpha2.TargetModel{
						{
							ModelRewrite: "model-a-old-tuned",
							Weight:       ptr.To[int32](100),
						},
					},
				},
			},
		},
	}

	rewriteNew := &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "rewrite-new",
			CreationTimestamp: metav1.Unix(2000, 0),
		},
		Spec: v1alpha2.InferenceModelRewriteSpec{
			Rules: []v1alpha2.InferenceModelRewriteRule{
				{
					Matches: []v1alpha2.Match{
						{
							Model: &v1alpha2.ModelMatch{
								Value: "model-a",
							},
						},
					},
					Targets: []v1alpha2.TargetModel{
						{
							ModelRewrite: "model-a-new-tuned",
							Weight:       ptr.To[int32](100),
						},
					},
				},
			},
		},
	}

	rewriteB := &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "rewrite-b",
			CreationTimestamp: metav1.Unix(1500, 0),
		},
		Spec: v1alpha2.InferenceModelRewriteSpec{
			Rules: []v1alpha2.InferenceModelRewriteRule{
				{
					Matches: []v1alpha2.Match{
						{
							Model: &v1alpha2.ModelMatch{
								Value: "model-b",
							},
						},
					},
					Targets: []v1alpha2.TargetModel{
						{
							ModelRewrite: "model-b-tuned",
							Weight:       ptr.To[int32](100),
						},
					},
				},
			},
		},
	}

	rewriteWeighted := &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "rewrite-weighted",
			CreationTimestamp: metav1.Unix(1200, 0),
		},
		Spec: v1alpha2.InferenceModelRewriteSpec{
			Rules: []v1alpha2.InferenceModelRewriteRule{
				{
					Matches: []v1alpha2.Match{
						{
							Model: &v1alpha2.ModelMatch{
								Value: "model-c",
							},
						},
					},
					Targets: []v1alpha2.TargetModel{
						{
							ModelRewrite: "model-c-v1",
							Weight:       ptr.To[int32](70),
						},
						{
							ModelRewrite: "model-c-v2",
							Weight:       ptr.To[int32](30),
						},
					},
				},
			},
		},
	}

	tests := []struct {
		name           string
		rewrites       []*v1alpha2.InferenceModelRewrite
		incomingModel  string
		expectedTarget []string
		initialTarget  string // Initial value of reqCtx.TargetModelName
	}{
		{
			name:           "no rewrites",
			rewrites:       []*v1alpha2.InferenceModelRewrite{},
			incomingModel:  "model-x",
			expectedTarget: []string{"model-x"},
			initialTarget:  "model-x",
		},
		{
			name:           "single matching rewrite",
			rewrites:       []*v1alpha2.InferenceModelRewrite{rewriteB},
			incomingModel:  "model-b",
			expectedTarget: []string{"model-b-tuned"},
			initialTarget:  "model-b",
		},
		{
			name:           "no matching rewrite",
			rewrites:       []*v1alpha2.InferenceModelRewrite{rewriteB},
			incomingModel:  "model-x",
			expectedTarget: []string{"model-x"},
			initialTarget:  "model-x",
		},
		{
			name:           "oldest rewrite wins for duplicate model",
			rewrites:       []*v1alpha2.InferenceModelRewrite{rewriteNew, rewriteOld}, // New is first, but Old has older timestamp
			incomingModel:  "model-a",
			expectedTarget: []string{"model-a-old-tuned"},
			initialTarget:  "model-a",
		},
		{
			name:           "weighted rewrite applied (probabilistic check)",
			rewrites:       []*v1alpha2.InferenceModelRewrite{rewriteWeighted},
			incomingModel:  "model-c",
			initialTarget:  "model-c",
			expectedTarget: []string{"model-c-v1", "model-c-v2"},
		},
		{
			name:           "initial TargetModelName is respected if no rewrite matches",
			rewrites:       []*v1alpha2.InferenceModelRewrite{rewriteB},
			incomingModel:  "model-x",
			initialTarget:  "pre-existing-target",
			expectedTarget: []string{"pre-existing-target"},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			mockDs := &mockDatastore{rewrites: test.rewrites}
			endpointCandidates := NewCachedEndpointCandidates(context.Background(), NewDatastoreEndpointCandidates(mockDs), time.Minute)
			director := NewDirectorWithConfig(mockDs, &mockScheduler{}, &mockAdmissionController{}, endpointCandidates, NewConfig())

			reqCtx := &handlers.RequestContext{
				IncomingModelName: test.incomingModel,
				TargetModelName:   test.initialTarget,
			}

			director.applyWeightedModelRewrite(t.Context(), reqCtx)
			assert.Contains(t, test.expectedTarget, reqCtx.TargetModelName, "TargetModelName mismatch")
		})
	}
}

func TestDirector_SelectWeightedModel(t *testing.T) {
	tests := []struct {
		name           string
		targets        []v1alpha2.TargetModel
		possibleModels sets.Set[string] // For probabilistic cases
	}{
		{
			name: "single target",
			targets: []v1alpha2.TargetModel{
				{ModelRewrite: "model-a", Weight: ptr.To[int32](100)},
			},
			possibleModels: sets.New("model-a"),
		},
		{
			name: "multiple targets, equal weight",
			targets: []v1alpha2.TargetModel{
				{ModelRewrite: "model-a", Weight: ptr.To[int32](50)},
				{ModelRewrite: "model-b", Weight: ptr.To[int32](50)},
			},
			possibleModels: sets.New("model-a", "model-b"),
		},
		{
			name: "multiple targets, different weights",
			targets: []v1alpha2.TargetModel{
				{ModelRewrite: "model-x", Weight: ptr.To[int32](70)},
				{ModelRewrite: "model-y", Weight: ptr.To[int32](30)},
			},
			possibleModels: sets.New("model-x", "model-y"),
		},
		{
			name: "omitted weights, distribute evenly",
			targets: []v1alpha2.TargetModel{
				{ModelRewrite: "model-z1"},
				{ModelRewrite: "model-z2"},
			},
			possibleModels: sets.New("model-z1", "model-z2"),
		},
		{
			name: "mixed weights select explicitly weighted targets",
			targets: []v1alpha2.TargetModel{
				{ModelRewrite: "model-without-weight"},
				{ModelRewrite: "model-with-weight", Weight: ptr.To[int32](100)},
			},
			possibleModels: sets.New("model-with-weight"),
		},
	}

	director := &Director{}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Run multiple times to check distribution
			counter := make(map[string]int)
			numRuns := 1000
			for range numRuns {
				selected := director.selectWeightedModel(t.Context(), test.targets)
				counter[selected]++
			}

			// Assert that all selected models are within the possible models
			for model := range counter {
				if !test.possibleModels.Has(model) {
					t.Errorf("Selected model %s is not in possible models %v", model, test.possibleModels)
				}
			}

			// Basic check for distribution (e.g., if 70/30, expect roughly 700/300)
			if len(test.targets) > 1 {
				totalWeight := int32(0)
				for _, target := range test.targets {
					if target.Weight != nil {
						totalWeight += *target.Weight
					}
				}

				if totalWeight == 0 { // Special case for zero total weight
					for _, target := range test.targets {
						expectedCount := numRuns / len(test.targets)
						assert.InDelta(t, expectedCount, counter[target.ModelRewrite], float64(numRuns)/float64(len(test.targets))*0.2, "Distribution for %s is off", target.ModelRewrite)
					}
				} else {
					for _, target := range test.targets {
						if target.Weight == nil {
							assert.Zero(t, counter[target.ModelRewrite], "Unweighted target %s should not be selected when other targets have weights", target.ModelRewrite)
							continue
						}
						expectedCount := float64(numRuns) * (float64(*target.Weight) / float64(totalWeight))
						assert.InDelta(t, expectedCount, float64(counter[target.ModelRewrite]), expectedCount*0.2, "Distribution for %s is off", target.ModelRewrite)
					}
				}
			}
		})
	}
}

func TestDirector_HandleResponseReceived(t *testing.T) {
	pr1 := newTestResponseReceived("pr1")

	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	ds := datastore.NewDatastore(t.Context(), nil)
	mockSched := &mockScheduler{}
	endpointCandidates := NewCachedEndpointCandidates(context.Background(), NewDatastoreEndpointCandidates(ds), time.Minute)
	director := NewDirectorWithConfig(
		ds,
		mockSched,
		&mockAdmissionController{},
		endpointCandidates,
		NewConfig().WithResponseReceivedPlugins(pr1),
	)

	reqCtx := &handlers.RequestContext{
		Request: &handlers.Request{
			Headers: map[string]string{
				reqcommon.RequestIDHeaderKey: "test-req-id-for-response",
			},
		},
		Response: &handlers.Response{ // Simulate some response headers
			Headers: map[string]string{"X-Test-Response-Header": "TestValue"},
		},

		TargetPod: &fwkdl.EndpointMetadata{NamespacedName: types.NamespacedName{Namespace: "namespace1", Name: "test-pod-name"}},
	}

	director.HandleResponseHeader(ctx, reqCtx)

	if diff := cmp.Diff("test-req-id-for-response", pr1.lastRespOnResponse.RequestID); diff != "" {
		t.Errorf("Scheduler.OnResponse RequestId mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff(reqCtx.Response.Headers, pr1.lastRespOnResponse.Headers); diff != "" {
		t.Errorf("Scheduler.OnResponse Headers mismatch (-want +got):\n%s", diff)
	}
	if diff := cmp.Diff("namespace1/test-pod-name", pr1.lastTargetPodOnResponse); diff != "" {
		t.Errorf("Scheduler.OnResponse TargetPodName mismatch (-want +got):\n%s", diff)
	}
}

// TestDirector_HandleResponseHeader_SessionAffinity validates that the
// session-affinity scorer and filter, registered as response-received plugins,
// set the session token header when the director runs the response-header hook.
func TestDirector_HandleResponseHeader_SessionAffinity(t *testing.T) {
	targetPod := &fwkdl.EndpointMetadata{NamespacedName: types.NamespacedName{Namespace: "namespace1", Name: "test-pod-name"}}
	wantToken := base64.StdEncoding.EncodeToString([]byte(targetPod.NamespacedName.String()))

	tests := []struct {
		name       string
		plugin     fwkrc.ResponseHeaderProcessor
		wantHeader string
	}{
		{
			name:       "scorer with default header",
			plugin:     sessionaffinityscorer.NewSessionAffinity("test-scorer", "", ""),
			wantHeader: "x-session-token",
		},
		{
			name:       "scorer with custom header",
			plugin:     sessionaffinityscorer.NewSessionAffinity("test-scorer", "x-custom-session", ""),
			wantHeader: "x-custom-session",
		},
		{
			name:       "filter with default header",
			plugin:     sessionaffinityfilter.NewSessionAffinity("test-filter", "", ""),
			wantHeader: "x-session-token",
		},
		{
			name:       "filter with custom header",
			plugin:     sessionaffinityfilter.NewSessionAffinity("test-filter", "x-custom-session", ""),
			wantHeader: "x-custom-session",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			ctx := logutil.NewTestLoggerIntoContext(context.Background())
			ds := datastore.NewDatastore(t.Context(), nil)
			endpointCandidates := NewCachedEndpointCandidates(context.Background(), NewDatastoreEndpointCandidates(ds), time.Minute)
			director := NewDirectorWithConfig(
				ds,
				&mockScheduler{},
				&mockAdmissionController{},
				endpointCandidates,
				NewConfig().WithResponseReceivedPlugins(test.plugin),
			)

			reqCtx := &handlers.RequestContext{
				Request: &handlers.Request{
					Headers: map[string]string{reqcommon.RequestIDHeaderKey: "test-req-id"},
				},
				Response:  &handlers.Response{Headers: map[string]string{}},
				TargetPod: targetPod,
			}

			director.HandleResponseHeader(ctx, reqCtx)

			assert.Equal(t, wantToken, reqCtx.Response.Headers[test.wantHeader],
				"session token should be set on the %q response header", test.wantHeader)
		})
	}
}

func TestDirector_HandleResponseBody(t *testing.T) {
	ps1 := newTestResponseStreaming("ps1")

	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	ds := datastore.NewDatastore(t.Context(), nil)
	mockSched := &mockScheduler{}
	endpointCandidates := NewCachedEndpointCandidates(context.Background(), NewDatastoreEndpointCandidates(ds), time.Minute)
	director := NewDirectorWithConfig(ds, mockSched, nil, endpointCandidates, NewConfig().WithResponseStreamingPlugins(ps1))

	reqCtx := &handlers.RequestContext{
		Request: &handlers.Request{
			Headers: map[string]string{
				reqcommon.RequestIDHeaderKey: "test-req-id-for-streaming",
			},
		},
		Response: &handlers.Response{
			Headers: map[string]string{"X-Test-Streaming-Header": "StreamValue"},
		},
		TargetPod: &fwkdl.EndpointMetadata{NamespacedName: types.NamespacedName{Namespace: "namespace1", Name: "test-pod-name"}},
	}

	director.HandleResponseBody(ctx, reqCtx, false)
	director.HandleResponseBody(ctx, reqCtx, false)

	// Intermediate chunks (endOfStream=false) run asynchronously, wait for them.
	require.Eventually(t, func() bool {
		ps1.mu.Lock()
		defer ps1.mu.Unlock()
		return len(ps1.respsOnStreaming) >= 2
	}, time.Second, 10*time.Millisecond, "async response body plugins should have been called for intermediate chunks")

	// Final chunk (endOfStream=true) runs synchronously (drains queue first).
	director.HandleResponseBody(ctx, reqCtx, true)

	ps1.mu.Lock()
	resps := make([]*fwkrc.Response, len(ps1.respsOnStreaming))
	copy(resps, ps1.respsOnStreaming)
	targetPods := make([]string, len(ps1.targetPodsOnStreaming))
	copy(targetPods, ps1.targetPodsOnStreaming)
	ps1.mu.Unlock()

	assert.Equal(t, 3, len(resps), "Should have received 3 streaming calls")

	for i, resp := range resps {
		assert.Equal(t, "test-req-id-for-streaming", resp.RequestID)
		assert.Equal(t, reqCtx.Response.Headers, resp.Headers)
		assert.Equal(t, "namespace1/test-pod-name", targetPods[i])
		if i < 2 {
			assert.False(t, resp.EndOfStream, "EndOfStream should be false for chunk %d", i)
		} else {
			assert.True(t, resp.EndOfStream, "EndOfStream should be true for last chunk")
		}
	}
}

func TestDirector_HandleResponseBody_ReleasesConcurrencyLease(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	ds := datastore.NewDatastore(t.Context(), nil, 0)
	mockSched := &mockScheduler{}
	endpointCandidates := NewCachedEndpointCandidates(context.Background(), NewDatastoreEndpointCandidates(ds), time.Minute)
	admissionController := &mockConcurrencyReleaserAdmissionController{}
	// No ResponseStreaming plugins configured: this specifically exercises the
	// path where HandleResponseBody's early return (no plugins registered)
	// would otherwise skip the release call entirely.
	director := NewDirectorWithConfig(ds, mockSched, admissionController, endpointCandidates, NewConfig())

	reqCtx := &handlers.RequestContext{
		SchedulingRequest: &fwksched.InferenceRequest{FairnessID: "tenant-a"},
		Priority:          3,
		Request: &handlers.Request{
			Headers: map[string]string{reqcommon.RequestIDHeaderKey: "test-req-release"},
		},
		Response: &handlers.Response{},
	}

	director.HandleResponseBody(ctx, reqCtx, false)
	assert.Zero(t, admissionController.releaseCalls, "must not release before EndOfStream")

	director.HandleResponseBody(ctx, reqCtx, true)
	require.Equal(t, 1, admissionController.releaseCalls)
	assert.Equal(t, "test-req-release", admissionController.lastRequestID)
	assert.Equal(t, "tenant-a", admissionController.lastFairness)
	assert.Equal(t, 3, admissionController.lastPriority)
}

func TestDirector_HandleResponseBody_ReleaseConcurrencyLeaseIsNoOpWithoutSchedulingRequest(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	ds := datastore.NewDatastore(t.Context(), nil, 0)
	mockSched := &mockScheduler{}
	endpointCandidates := NewCachedEndpointCandidates(context.Background(), NewDatastoreEndpointCandidates(ds), time.Minute)
	admissionController := &mockConcurrencyReleaserAdmissionController{}
	director := NewDirectorWithConfig(ds, mockSched, admissionController, endpointCandidates, NewConfig())

	reqCtx := &handlers.RequestContext{
		Request:  &handlers.Request{Headers: map[string]string{reqcommon.RequestIDHeaderKey: "test-req"}},
		Response: &handlers.Response{},
	}

	director.HandleResponseBody(ctx, reqCtx, true) // must not panic despite nil SchedulingRequest

	assert.Zero(t, admissionController.releaseCalls)
}

func TestDirector_HandleResponseBody_ChunkOrdering(t *testing.T) {
	// orderTrackingPlugin records the RequestId of each chunk it processes.
	// Since we set a unique RequestId per chunk, the recorded order lets us
	// verify that chunks are processed in the exact order they were sent,
	// even though they go through the async queue.
	plugin := &orderTrackingPlugin{
		typedName: fwkplugin.TypedName{Type: "order-tracker", Name: "order-tracker"},
	}

	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	ds := datastore.NewDatastore(t.Context(), nil)
	director := NewDirectorWithConfig(ds, &mockScheduler{}, nil, nil, NewConfig().WithResponseStreamingPlugins(plugin))

	const numChunks = 50
	reqCtx := newResponseBodyTestRequestContext("ordering-test-request", 0)

	for i := range numChunks {
		reqCtx.Usage = fwkrh.Usage{CompletionTokens: i}
		director.HandleResponseBody(ctx, reqCtx, false)
	}

	// Send final chunk to drain the queue.
	reqCtx.Usage = fwkrh.Usage{CompletionTokens: numChunks}
	director.HandleResponseBody(ctx, reqCtx, true)

	// Total calls: numChunks async + 1 sync final.
	plugin.mu.Lock()
	tokenCounts := make([]int, len(plugin.observedTokenCounts))
	copy(tokenCounts, plugin.observedTokenCounts)
	plugin.mu.Unlock()

	require.Equal(t, numChunks+1, len(tokenCounts), "should have received all chunk calls")

	// Verify ordering: each chunk's CompletionTokens should appear in the order 0, 1, 2, ..., numChunks.
	for i, tokens := range tokenCounts {
		assert.Equal(t, i, tokens, "chunk %d was processed out of order", i)
	}
}

func TestDirector_HandleResponseBody_DuplicateRequestIDQueuesAreIndependent(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	plugin := newBlockingResponseStreamingPlugin()
	director := NewDirectorWithConfig(nil, &mockScheduler{}, nil, nil, NewConfig().WithResponseStreamingPlugins(plugin))

	const requestID = "duplicate-request-id"
	firstReqCtx := newResponseBodyTestRequestContext(requestID, 0)
	secondReqCtx := newResponseBodyTestRequestContext(requestID, 0)

	director.HandleResponseBody(ctx, firstReqCtx, false)
	require.Eventually(t, func() bool {
		return plugin.started()
	}, time.Second, 10*time.Millisecond, "first request should start processing")

	for i := range responseBodyQueueCapacity {
		firstReqCtx.Usage = fwkrh.Usage{CompletionTokens: i + 1}
		director.HandleResponseBody(ctx, firstReqCtx, false)
	}

	secondDone := make(chan any, 1)
	go func() {
		defer func() {
			secondDone <- recover()
		}()
		director.HandleResponseBody(ctx, secondReqCtx, false)
	}()

	secondCompletedBeforeFinal := false
	select {
	case panicValue := <-secondDone:
		require.Nil(t, panicValue, "second request with duplicate request ID should not panic")
		secondCompletedBeforeFinal = true
	case <-time.After(time.Second):
	}

	firstFinalDone := make(chan any, 1)
	go func() {
		defer func() {
			firstFinalDone <- recover()
		}()
		director.HandleResponseBody(ctx, firstReqCtx, true)
	}()

	if !secondCompletedBeforeFinal {
		select {
		case panicValue := <-secondDone:
			require.Nil(t, panicValue, "second request with duplicate request ID should not panic")
		case <-time.After(time.Second):
			t.Fatal("second request with duplicate request ID should not remain blocked")
		}
	}
	require.True(t, secondCompletedBeforeFinal, "second request with duplicate request ID should not block behind the first request queue")

	plugin.release()

	select {
	case panicValue := <-firstFinalDone:
		require.Nil(t, panicValue, "first request final chunk should not panic")
	case <-time.After(time.Second):
		t.Fatal("first request final chunk should drain")
	}

	director.HandleResponseBody(ctx, secondReqCtx, true)
}

// orderTrackingPlugin records the CompletionTokens from each ResponseBody call to verify ordering.
type orderTrackingPlugin struct {
	mu                  sync.Mutex
	typedName           fwkplugin.TypedName
	observedTokenCounts []int
}

func (p *orderTrackingPlugin) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *orderTrackingPlugin) ResponseBody(_ context.Context, _ *fwksched.InferenceRequest, response *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.observedTokenCounts = append(p.observedTokenCounts, response.Usage.CompletionTokens)
}

const (
	testResponseReceivedType = "test-response-received"
	testPostStreamingType    = "test-response-streaming"
	testPostCompleteType     = "test-response-complete"
)

type testResponseReceived struct {
	mu                      sync.Mutex
	typedName               fwkplugin.TypedName
	lastRespOnResponse      *fwkrc.Response
	lastTargetPodOnResponse string
}

type testResponseStreaming struct {
	mu                    sync.Mutex
	typedName             fwkplugin.TypedName
	respsOnStreaming      []*fwkrc.Response
	targetPodsOnStreaming []string

	// Legacy fields for existing tests if any, but better to update them
	lastRespOnStreaming      *fwkrc.Response
	lastTargetPodOnStreaming string
}

func newTestResponseReceived(name string) *testResponseReceived {
	return &testResponseReceived{
		typedName: fwkplugin.TypedName{Type: testResponseReceivedType, Name: name},
	}
}

func newTestResponseStreaming(name string) *testResponseStreaming {
	return &testResponseStreaming{
		typedName: fwkplugin.TypedName{Type: testPostStreamingType, Name: name},
	}
}

func (p *testResponseReceived) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *testResponseStreaming) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *testResponseReceived) ResponseHeader(_ context.Context, _ *fwksched.InferenceRequest, response *fwkrc.Response, targetPod *fwkdl.EndpointMetadata) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.lastRespOnResponse = response
	p.lastTargetPodOnResponse = targetPod.NamespacedName.String()
}

func (p *testResponseStreaming) ResponseBody(_ context.Context, _ *fwksched.InferenceRequest, response *fwkrc.Response, targetPod *fwkdl.EndpointMetadata) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.respsOnStreaming = append(p.respsOnStreaming, response)
	p.targetPodsOnStreaming = append(p.targetPodsOnStreaming, targetPod.NamespacedName.String())

	// Maintain legacy fields for compatibility
	p.lastRespOnStreaming = response
	p.lastTargetPodOnStreaming = targetPod.NamespacedName.String()
}

func TestResponseBodyQueue_CloseWaitsForBlockedEnqueue(t *testing.T) {
	q := newResponseBodyQueue()
	close(q.done)

	for range responseBodyQueueCapacity {
		require.True(t, q.enqueue(responseBodyWork{}))
	}

	enqueueDone := make(chan any, 1)
	go func() {
		defer func() {
			enqueueDone <- recover()
		}()
		q.enqueue(responseBodyWork{})
	}()

	require.Eventually(t, func() bool {
		if q.mu.TryLock() {
			q.mu.Unlock()
			return false
		}
		return true
	}, time.Second, 10*time.Millisecond, "enqueue should block while the queue is full")

	closeDone := make(chan any, 1)
	go func() {
		defer func() {
			closeDone <- recover()
		}()
		q.closeAndWait()
	}()

	<-q.ch

	select {
	case panicValue := <-enqueueDone:
		require.Nil(t, panicValue, "enqueue should not panic when close waits")
	case <-time.After(time.Second):
		t.Fatal("enqueue should finish after queue space is available")
	}

	select {
	case panicValue := <-closeDone:
		require.Nil(t, panicValue, "close should not panic")
	case <-time.After(time.Second):
		t.Fatal("close should finish after enqueue completes")
	}

	require.False(t, q.enqueue(responseBodyWork{}), "enqueue should fail after the queue is closed")
}

type blockingResponseStreamingPlugin struct {
	typedName fwkplugin.TypedName
	once      sync.Once
	startedCh chan struct{}
	releaseCh chan struct{}
}

func newBlockingResponseStreamingPlugin() *blockingResponseStreamingPlugin {
	return &blockingResponseStreamingPlugin{
		typedName: fwkplugin.TypedName{Type: testPostStreamingType, Name: "blocking"},
		startedCh: make(chan struct{}),
		releaseCh: make(chan struct{}),
	}
}

func (p *blockingResponseStreamingPlugin) TypedName() fwkplugin.TypedName {
	return p.typedName
}

func (p *blockingResponseStreamingPlugin) ResponseBody(_ context.Context, _ *fwksched.InferenceRequest, _ *fwkrc.Response, _ *fwkdl.EndpointMetadata) {
	p.once.Do(func() {
		close(p.startedCh)
	})
	<-p.releaseCh
}

func (p *blockingResponseStreamingPlugin) started() bool {
	select {
	case <-p.startedCh:
		return true
	default:
		return false
	}
}

func (p *blockingResponseStreamingPlugin) release() {
	close(p.releaseCh)
}

func newResponseBodyTestRequestContext(requestID string, completionTokens int) *handlers.RequestContext {
	return &handlers.RequestContext{
		Request: &handlers.Request{
			Headers: map[string]string{
				reqcommon.RequestIDHeaderKey: requestID,
			},
		},
		Response: &handlers.Response{
			Headers: map[string]string{},
		},
		TargetPod: &fwkdl.EndpointMetadata{},
		Usage:     fwkrh.Usage{CompletionTokens: completionTokens},
	}
}

// ── Conditional-decode gate (Prefer: if-available) ─────────────────────────

// wrongTypeAttr is a Cloneable that is NOT *attrprefix.PrefixCacheMatchInfo,
// used to exercise the type-assertion failure branch.
type wrongTypeAttr struct{}

func (w wrongTypeAttr) Clone() fwkdl.Cloneable { return w }

func TestPrimaryEndpointHasCachedPrefix(t *testing.T) {
	endpointWith := func(matched, total int) fwksched.Endpoint {
		attrs := fwkdl.NewAttributes()
		attrs.Put(attrprefix.PrefixCacheMatchInfoDataKey.String(),
			attrprefix.NewPrefixCacheMatchInfo(matched, total, 1))
		return fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{NamespacedName: types.NamespacedName{Namespace: "default", Name: "p"}},
			nil, attrs,
		)
	}
	endpointBare := func() fwksched.Endpoint {
		return fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{NamespacedName: types.NamespacedName{Namespace: "default", Name: "p"}},
			nil, fwkdl.NewAttributes(),
		)
	}
	endpointWithWrongType := func() fwksched.Endpoint {
		attrs := fwkdl.NewAttributes()
		attrs.Put(attrprefix.PrefixCacheMatchInfoDataKey.String(), wrongTypeAttr{})
		return fwksched.NewEndpoint(
			&fwkdl.EndpointMetadata{NamespacedName: types.NamespacedName{Namespace: "default", Name: "p"}},
			nil, attrs,
		)
	}
	resultWith := func(eps ...fwksched.Endpoint) *fwksched.SchedulingResult {
		return &fwksched.SchedulingResult{
			PrimaryProfileName: "decode",
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"decode": {TargetEndpoints: eps},
			},
		}
	}

	tests := []struct {
		name string
		in   *fwksched.SchedulingResult
		want bool
	}{
		{"nil result", nil, false},
		{"empty profile results", &fwksched.SchedulingResult{PrimaryProfileName: "decode"}, false},
		{"primary profile missing", &fwksched.SchedulingResult{
			PrimaryProfileName: "decode",
			ProfileResults:     map[string]*fwksched.ProfileRunResult{"other": {TargetEndpoints: []fwksched.Endpoint{endpointWith(2, 4)}}},
		}, false},
		{"primary profile nil", &fwksched.SchedulingResult{
			PrimaryProfileName: "decode",
			ProfileResults:     map[string]*fwksched.ProfileRunResult{"decode": nil},
		}, false},
		{"primary has no endpoints", resultWith(), false},
		{"endpoint has no match info", resultWith(endpointBare()), false},
		{"wrong type in attribute", resultWith(endpointWithWrongType()), false},
		{"zero match blocks", resultWith(endpointWith(0, 4)), false},
		{"some match blocks", resultWith(endpointWith(2, 4)), true},
		{"full match", resultWith(endpointWith(4, 4)), true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, primaryEndpointHasCachedPrefix(logr.Discard(), tt.in))
		})
	}
}

// newConditionalDecodeDirector builds a minimal Director suitable for
// exercising the conditional-decode gate end-to-end.
func newConditionalDecodeDirector(t *testing.T, scheduleResult *fwksched.SchedulingResult) (*Director, context.Context) {
	t.Helper()
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	period := time.Second
	epf := datalayer.NewTestRuntime(t, period)
	ds := datastore.NewDatastore(t.Context(), epf)
	scheme := runtime.NewScheme()
	_ = clientgoscheme.AddToScheme(scheme)
	fakeClient := fake.NewClientBuilder().WithScheme(scheme).Build()
	pool := &v1.InferencePool{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pool", Namespace: "default"},
		Spec: v1.InferencePoolSpec{
			TargetPorts: []v1.Port{{Number: v1.PortNumber(int32(8000))}},
			Selector: v1.LabelSelector{
				MatchLabels: map[v1.LabelKey]v1.LabelValue{"app": "inference"},
			},
		},
	}
	if err := ds.PoolSet(ctx, fakeClient, poolutil.InferencePoolToEndpointPool(pool)); err != nil {
		t.Fatalf("PoolSet: %v", err)
	}
	ds.PodUpdateOrAddIfNotExist(ctx, &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: "default", Labels: map[string]string{"app": "inference"}},
		Status: corev1.PodStatus{
			PodIP:      "192.168.1.100",
			Phase:      corev1.PodRunning,
			Conditions: []corev1.PodCondition{{Type: corev1.PodReady, Status: corev1.ConditionTrue}},
		},
	})

	mockSched := &mockScheduler{scheduleResults: scheduleResult}
	cfg := NewConfig().WithAdmissionPlugins(newMockAdmissionPlugin("admit", nil))
	candidates := NewCachedEndpointCandidates(context.Background(), NewDatastoreEndpointCandidates(ds), time.Minute)
	dir := NewDirectorWithConfig(ds, mockSched, &mockAdmissionController{}, candidates, cfg)
	return dir, ctx
}

func TestDirector_HandleRequest_ConditionalDecode(t *testing.T) {
	scheduleResultWith := func(matched, total int) *fwksched.SchedulingResult {
		attrs := fwkdl.NewAttributes()
		if matched >= 0 {
			attrs.Put(attrprefix.PrefixCacheMatchInfoDataKey.String(),
				attrprefix.NewPrefixCacheMatchInfo(matched, total, 1))
		}
		return &fwksched.SchedulingResult{
			PrimaryProfileName: "decode",
			ProfileResults: map[string]*fwksched.ProfileRunResult{
				"decode": {TargetEndpoints: []fwksched.Endpoint{
					fwksched.NewEndpoint(&fwkdl.EndpointMetadata{
						Address:        "192.168.1.100",
						Port:           "8000",
						MetricsHost:    "192.168.1.100:8000",
						NamespacedName: types.NamespacedName{Name: "pod1", Namespace: "default"},
					}, nil, attrs),
				}},
			},
		}
	}

	tests := []struct {
		name        string
		preferValue string // empty == no Prefer header
		matched     int    // -1 == no PrefixCacheMatchInfo at all
		total       int
		wantErrCode string
	}{
		{"prefer if-available + cache hit forwards", "if-available", 2, 4, ""},
		{"prefer if-available + zero match returns 412", "if-available", 0, 4, errcommon.PreconditionFailed},
		{"prefer if-available + no match info returns 412", "if-available", -1, 0, errcommon.PreconditionFailed},
		{"absent Prefer header proceeds even with no cache", "", -1, 0, ""},
		{"unrelated Prefer token proceeds even with no cache", "return=minimal", -1, 0, ""},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dir, ctx := newConditionalDecodeDirector(t, scheduleResultWith(tt.matched, tt.total))

			reqCtx := &handlers.RequestContext{
				Request: &handlers.Request{
					Headers: map[string]string{
						reqcommon.RequestIDHeaderKey: "test-req-id",
						":path":                      "/v1/completions",
					},
				},
			}
			if tt.preferValue != "" {
				reqCtx.Request.Headers["prefer"] = tt.preferValue
			}
			body, err := json.Marshal(map[string]any{"model": "m", "prompt": "p"})
			require.NoError(t, err)
			reqCtx.Request.RawBody = body

			parseResult, err := openai.NewOpenAIParser().ParseRequest(ctx, reqCtx.Request.RawBody, reqCtx.Request.Headers)
			require.NoError(t, err)

			_, err = dir.HandleRequest(ctx, reqCtx, parseResult.Body)
			if tt.wantErrCode == "" {
				assert.NoError(t, err)
				return
			}
			require.Error(t, err)
			var e errcommon.Error
			require.ErrorAs(t, err, &e)
			assert.Equal(t, tt.wantErrCode, e.Code)
		})
	}
}
