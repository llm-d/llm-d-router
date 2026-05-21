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

package requesthandler

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	v1 "sigs.k8s.io/gateway-api-inference-extension/api/v1"

	"github.com/llm-d/llm-d-router/apix/v1alpha2"
	errcommon "github.com/llm-d/llm-d-router/pkg/common/error"
	logutil "github.com/llm-d/llm-d-router/pkg/common/observability/logging"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	backendmetrics "github.com/llm-d/llm-d-router/pkg/epp/backend/metrics"
	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/datastore"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/requesthandler/parsers/openai"
	fwkrhapi "github.com/llm-d/llm-d-router/pkg/epp/framework/requesthandler/types"
	"github.com/llm-d/llm-d-router/pkg/epp/handlers"
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

type mockScheduler struct {
	scheduleResults *fwkrhapi.SchedulingResult
	scheduleErr     error
	dataProduced    bool // denotes whether data production is expected.
}

func (m *mockScheduler) Schedule(_ context.Context, _ *fwkrhapi.InferenceRequest, endpoints []fwkrhapi.Endpoint) (*fwkrhapi.SchedulingResult, error) {
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
		if predicate == nil || predicate(pod) {
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

func (m *mockDataProducerPlugin) Consumes() map[fwkplugin.DataKey]any {
	return m.consumes
}

func (m *mockDataProducerPlugin) Produce(ctx context.Context, request *fwkrhapi.InferenceRequest, endpoints []fwkrhapi.Endpoint) error {
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

func (m *mockAdmissionPlugin) AdmitRequest(ctx context.Context, request *fwkrhapi.InferenceRequest, endpoints []fwkrhapi.Endpoint) error {
	return m.denialError
}

type mockPreRequestPlugin struct {
	name     string
	modifyFn func(request *fwkrhapi.InferenceRequest)
}

func (m *mockPreRequestPlugin) TypedName() fwkplugin.TypedName {
	return fwkplugin.TypedName{Name: m.name, Type: "mock"}
}

func (m *mockPreRequestPlugin) PreRequest(ctx context.Context, request *fwkrhapi.InferenceRequest, schedulingResult *fwkrhapi.SchedulingResult) {
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
	var matchingRewrites []*v1alpha2.InferenceModelRewrite
	for _, r := range ds.rewrites {
		for _, rule := range r.Spec.Rules {
			for _, match := range rule.Matches {
				if match.Model != nil && match.Model.Value == modelName {
					matchingRewrites = append(matchingRewrites, r)
					break
				}
			}
		}
	}

	if len(matchingRewrites) == 0 {
		return nil, ""
	}

	sort.Slice(matchingRewrites, func(i, j int) bool {
		return matchingRewrites[i].CreationTimestamp.Before(&matchingRewrites[j].CreationTimestamp)
	})

	return &matchingRewrites[0].Spec.Rules[0], matchingRewrites[0].Name
}

func TestDirector_HandleRequest(t *testing.T) {
	ctx := logutil.NewTestLoggerIntoContext(context.Background())

	model := "food-review"
	modelToBeRewritten := "food-review-to-be-rewritten"
	modelRewritten := "food-review-rewritten"

	objectiveName := "ioFoodReview"

	ioFoodReview := testutil.MakeInferenceObjective("ioFoodReview").CreationTimestamp(metav1.Unix(1000, 0)).Priority(2).ObjRef()
	ioFoodReviewSheddable := testutil.MakeInferenceObjective("imFoodReviewSheddable").CreationTimestamp(metav1.Unix(1000, 0)).Priority(-1).ObjRef()
	ioFoodReviewResolve := testutil.MakeInferenceObjective("imFoodReviewResolve").CreationTimestamp(metav1.Unix(1000, 0)).Priority(1).ObjRef()

	rewrite := &v1alpha2.InferenceModelRewrite{
		ObjectMeta: metav1.ObjectMeta{Name: "rewrite-rule", CreationTimestamp: metav1.Now()},
		Spec: v1alpha2.InferenceModelRewriteSpec{
			Rules: []v1alpha2.InferenceModelRewriteRule{
				{
					Matches: []v1alpha2.Match{{Model: &v1alpha2.ModelMatch{Value: modelToBeRewritten}}},
					Targets: []v1alpha2.TargetModel{{ModelRewrite: modelRewritten, Weight: 100}},
				},
			},
		},
	}

	pool := &v1.InferencePool{
		ObjectMeta: metav1.ObjectMeta{Name: "test-pool", Namespace: "default"},
		Spec: v1.InferencePoolSpec{
			TargetPorts: []v1.Port{{Number: v1.PortNumber(int32(8000))}},
			Selector: v1.LabelSelector{MatchLabels: map[v1.LabelKey]v1.LabelValue{"app": "inference"}},
		},
	}

	defaultSuccessfulScheduleResults := &fwkrhapi.SchedulingResult{
		ProfileResults: map[string]*fwkrhapi.ProfileRunResult{
			"testProfile": {
				TargetEndpoints: []fwkrhapi.Endpoint{
					fwkrhapi.NewEndpoint(&fwkdl.EndpointMetadata{
						Address:        "192.168.1.100",
						Port:           "8000",
						MetricsHost:    "192.168.1.100:8000",
						NamespacedName: types.NamespacedName{Name: "pod1", Namespace: "default"},
					}, nil, nil),
					fwkrhapi.NewEndpoint(&fwkdl.EndpointMetadata{
						Address:        "192.168.2.100",
						Port:           "8000",
						MetricsHost:    "192.168.2.100:8000",
						NamespacedName: types.NamespacedName{Name: "pod2", Namespace: "default"},
					}, nil, nil),
					fwkrhapi.NewEndpoint(&fwkdl.EndpointMetadata{
						Address:        "192.168.4.100",
						Port:           "8000",
						MetricsHost:    "192.168.4.100:8000",
						NamespacedName: types.NamespacedName{Name: "pod4", Namespace: "default"},
					}, nil, nil),
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
		initialTargetModelName  string
		parser                  fwkrhapi.Parser
		wantErrCode             string
		wantReqCtx              *handlers.RequestContext
		targetModelName         string
		admitRequestDenialError error
		dataProducerPlugin      *mockDataProducerPlugin
		preRequestPlugin        *mockPreRequestPlugin
		wantMutatedBody         map[string]any
	}{
		{
			name: "successful completions request",
			reqBodyMap: map[string]any{"model": model, "prompt": "critical prompt"},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) { m.scheduleResults = defaultSuccessfulScheduleResults },
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
			wantMutatedBody: map[string]any{"model": model, "prompt": "critical prompt"},
			inferenceObjectiveName: objectiveName,
		},
		{
			name: "successful request with preRequest plugin adding key",
			reqBodyMap: map[string]any{"model": model, "prompt": "original prompt"},
			mockAdmissionController: &mockAdmissionController{admitErr: nil},
			schedulerMockSetup: func(m *mockScheduler) { m.scheduleResults = defaultSuccessfulScheduleResults },
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
			wantMutatedBody: map[string]any{"model": model, "prompt": "original prompt", "new_key": "new_value"},
			inferenceObjectiveName: objectiveName,
			preRequestPlugin: &mockPreRequestPlugin{
				name: "test-pre-request-plugin",
				modifyFn: func(request *fwkrhapi.InferenceRequest) {
					if payloadMap, ok := request.Body.Payload.(fwkrhapi.PayloadMap); ok {
						payloadMap["new_key"] = "new_value"
					}
				},
			},
		},
	}

	period := time.Second
	factories := []datalayer.EndpointFactory{
		backendmetrics.NewPodMetricsFactory(&backendmetrics.FakePodMetricsClient{}, period),
		datalayer.NewTestRuntime(t, period),
	}
	for _, epf := range factories {
		ds := datastore.NewDatastore(t.Context(), epf, 0)
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

				reqCtx := &handlers.RequestContext{
					Request: &handlers.Request{
						Headers: map[string]string{
							reqcommon.RequestIDHeaderKey: "test-req-id-" + test.name,
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

				if _, hasPrompt := test.reqBodyMap["prompt"]; hasPrompt {
					reqCtx.Request.Headers[":path"] = "/v1/completions"
				} else if _, hasMessages := test.reqBodyMap["messages"]; hasMessages {
					reqCtx.Request.Headers[":path"] = "/v1/chat/completions"
				}

				parseResult, parseErr := openai.NewOpenAIParser().ParseRequest(ctx, reqCtx.Request.RawBody, reqCtx.Request.Headers)
				var returnedReqCtx *handlers.RequestContext
				if parseErr != nil {
					err = errcommon.Error{Code: errcommon.BadRequest, Msg: parseErr.Error()}
				} else {
					returnedReqCtx, err = director.HandleRequest(ctx, reqCtx, parseResult.Body)
				}

				if test.wantErrCode != "" {
					assert.Error(t, err)
					return
				}

				assert.NoError(t, err)
				if test.wantReqCtx != nil {
					assert.Equal(t, test.wantReqCtx.TargetModelName, returnedReqCtx.TargetModelName)
				}
			})
		}
	}
}

func TestDirector_HandleResponseHeader(t *testing.T) {
	pr1 := newTestResponseReceived("pr1")
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	ds := datastore.NewDatastore(t.Context(), nil, 0)
	mockSched := &mockScheduler{}
	endpointCandidates := NewCachedEndpointCandidates(context.Background(), NewDatastoreEndpointCandidates(ds), time.Minute)
	director := NewDirectorWithConfig(ds, mockSched, &mockAdmissionController{}, endpointCandidates, NewConfig().WithResponseReceivedPlugins(pr1))

	reqCtx := &handlers.RequestContext{
		Request: &handlers.Request{
			Headers: map[string]string{
				reqcommon.RequestIDHeaderKey: "test-req-id-for-response",
			},
		},
		Response: &handlers.Response{
			Headers: map[string]string{"X-Test-Response-Header": "TestValue"},
		},
		TargetPod: &fwkdl.EndpointMetadata{NamespacedName: types.NamespacedName{Namespace: "namespace1", Name: "test-pod-name"}},
	}

	director.HandleResponseHeader(ctx, reqCtx)

	assert.Equal(t, "test-req-id-for-response", pr1.lastRespOnResponse.RequestID)
	assert.Equal(t, reqCtx.Response.Headers, pr1.lastRespOnResponse.Headers)
}

func TestDirector_HandleResponseBody(t *testing.T) {
	ps1 := newTestResponseStreaming("ps1")
	ctx := logutil.NewTestLoggerIntoContext(context.Background())
	ds := datastore.NewDatastore(t.Context(), nil, 0)
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
	director.HandleResponseBody(ctx, reqCtx, true)

	require.Eventually(t, func() bool {
		ps1.mu.Lock()
		defer ps1.mu.Unlock()
		return len(ps1.respsOnStreaming) >= 2
	}, time.Second, 10*time.Millisecond)

	assert.Equal(t, 2, len(ps1.respsOnStreaming))
}

type orderTrackingPlugin struct {
	mu                  sync.Mutex
	typedName           fwkplugin.TypedName
	observedTokenCounts []int
}

func (p *orderTrackingPlugin) TypedName() fwkplugin.TypedName { return p.typedName }
func (p *orderTrackingPlugin) ResponseBody(_ context.Context, _ *fwkrhapi.InferenceRequest, response *fwkrhapi.Response, _ *fwkdl.EndpointMetadata) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.observedTokenCounts = append(p.observedTokenCounts, response.Usage.CompletionTokens)
}

type testResponseReceived struct {
	mu                      sync.Mutex
	typedName               fwkplugin.TypedName
	lastRespOnResponse      *fwkrhapi.Response
	lastTargetPodOnResponse string
}

func (p *testResponseReceived) TypedName() fwkplugin.TypedName { return p.typedName }
func (p *testResponseReceived) ResponseHeader(_ context.Context, _ *fwkrhapi.InferenceRequest, response *fwkrhapi.Response, targetPod *fwkdl.EndpointMetadata) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.lastRespOnResponse = response
	p.lastTargetPodOnResponse = targetPod.NamespacedName.String()
}

type testResponseStreaming struct {
	mu               sync.Mutex
	typedName        fwkplugin.TypedName
	respsOnStreaming []*fwkrhapi.Response
}

func (p *testResponseStreaming) TypedName() fwkplugin.TypedName { return p.typedName }
func (p *testResponseStreaming) ResponseBody(_ context.Context, _ *fwkrhapi.InferenceRequest, response *fwkrhapi.Response, _ *fwkdl.EndpointMetadata) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.respsOnStreaming = append(p.respsOnStreaming, response)
}

func newTestResponseReceived(name string) *testResponseReceived {
	return &testResponseReceived{typedName: fwkplugin.TypedName{Type: "test-recv", Name: name}}
}

func newTestResponseStreaming(name string) *testResponseStreaming {
	return &testResponseStreaming{typedName: fwkplugin.TypedName{Type: "test-stream", Name: name}}
}

func newResponseBodyTestRequestContext(requestID string, completionTokens int) *handlers.RequestContext {
	return &handlers.RequestContext{
		Request:   &handlers.Request{Headers: map[string]string{reqcommon.RequestIDHeaderKey: requestID}},
		Response:  &handlers.Response{Headers: map[string]string{}},
		TargetPod: &fwkdl.EndpointMetadata{},
		Usage:     fwkrhapi.Usage{CompletionTokens: completionTokens},
	}
}

func newBlockingResponseStreamingPlugin() *blockingResponseStreamingPlugin {
	return &blockingResponseStreamingPlugin{
		typedName: fwkplugin.TypedName{Type: "blocking", Name: "blocking"},
		startedCh: make(chan struct{}),
		releaseCh: make(chan struct{}),
	}
}

type blockingResponseStreamingPlugin struct {
	typedName fwkplugin.TypedName
	once      sync.Once
	startedCh chan struct{}
	releaseCh chan struct{}
}

func (p *blockingResponseStreamingPlugin) TypedName() fwkplugin.TypedName { return p.typedName }
func (p *blockingResponseStreamingPlugin) ResponseBody(_ context.Context, _ *fwkrhapi.InferenceRequest, _ *fwkrhapi.Response, _ *fwkdl.EndpointMetadata) {
	p.once.Do(func() { close(p.startedCh) })
	<-p.releaseCh
}
func (p *blockingResponseStreamingPlugin) started() bool {
	select {
	case <-p.startedCh: return true
	default: return false
	}
}
func (p *blockingResponseStreamingPlugin) release() { close(p.releaseCh) }
