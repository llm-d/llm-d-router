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

package datalayer

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/go-logr/logr"
	promtestutil "github.com/prometheus/client_golang/prometheus/testutil"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/workqueue"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	extractormocks "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/mocks"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
	"github.com/llm-d/llm-d-router/pkg/epp/metrics"
)

type testSyncingSource struct {
	waitErr error
}

func (*testSyncingSource) Start(context.Context, workqueue.TypedRateLimitingInterface[reconcile.Request]) error {
	return nil
}

func (s *testSyncingSource) WaitForSync(context.Context) error {
	return s.waitErr
}

func TestNotificationInitialSync(t *testing.T) {
	initialSync := newNotificationInitialSync("pods")
	first := types.NamespacedName{Namespace: "default", Name: "first"}
	second := types.NamespacedName{Namespace: "default", Name: "second"}
	initialSync.tracker.Start(first)
	initialSync.tracker.Start(second)

	wrapped := &notificationInitialSyncSource{
		SyncingSource: &testSyncingSource{},
		initialSync:   initialSync,
	}
	if err := wrapped.WaitForSync(context.Background()); err != nil {
		t.Fatalf("WaitForSync: %v", err)
	}
	if initialSync.hasSynced() {
		t.Fatal("initial sync completed before queued keys were processed")
	}

	initialSync.tracker.Finished(first)
	if initialSync.hasSynced() {
		t.Fatal("initial sync completed before every queued key was processed")
	}

	initialSync.tracker.Finished(second)
	if !initialSync.hasSynced() {
		t.Fatal("initial sync did not complete after every queued key was processed")
	}
}

func TestNotificationInitialSyncWaitError(t *testing.T) {
	wantErr := errors.New("sync failed")
	initialSync := newNotificationInitialSync("pods")
	wrapped := &notificationInitialSyncSource{
		SyncingSource: &testSyncingSource{waitErr: wantErr},
		initialSync:   initialSync,
	}
	if err := wrapped.WaitForSync(context.Background()); !errors.Is(err, wantErr) {
		t.Fatalf("WaitForSync() error = %v, want %v", err, wantErr)
	}
	if initialSync.hasSynced() {
		t.Fatal("initial sync completed after the upstream source failed to sync")
	}
}

func TestRuntimeNotificationReadiness(t *testing.T) {
	runtime := NewRuntime(time.Second)
	if err := runtime.CheckReady(); err == nil {
		t.Fatal("CheckReady() before Start returned nil")
	}

	if err := runtime.Start(context.Background(), nil); err != nil {
		t.Fatalf("Start() without notification sources: %v", err)
	}
	if err := runtime.CheckReady(); err != nil {
		t.Fatalf("CheckReady() without notification sources: %v", err)
	}

	initialSync := newNotificationInitialSync("pods")
	runtime.notificationSyncs = []*notificationInitialSync{initialSync}
	if err := runtime.CheckReady(); err == nil || !strings.Contains(err.Error(), "pods") {
		t.Fatalf("CheckReady() error = %v, want pending source name", err)
	}

	initialSync.tracker.UpstreamHasSynced()
	if err := runtime.CheckReady(); err != nil {
		t.Fatalf("CheckReady() after an empty initial list: %v", err)
	}
}

func TestNotificationDispatchReturnsExtractorErrors(t *testing.T) {
	wantErr := errors.New("extract failed")
	gvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}
	failing := extractormocks.NewNotificationExtractor("failing").WithExtractError(wantErr)
	working := extractormocks.NewNotificationExtractor("working")
	reconciler := &notificationReconciler{
		src: sourcenotifications.NewK8sNotificationSource(
			sourcenotifications.NotificationSourceType,
			"pods",
			gvk,
		),
		extractors: []fwkdl.NotificationExtractor{failing, working},
		log:        logr.Discard(),
	}
	event := &fwkdl.NotificationEvent{Object: &unstructured.Unstructured{}}

	if err := reconciler.dispatch(context.Background(), logr.Discard(), event); !errors.Is(err, wantErr) {
		t.Fatalf("dispatch() error = %v, want %v", err, wantErr)
	}
	if got := len(working.GetEvents()); got != 1 {
		t.Fatalf("working extractor received %d events, want 1", got)
	}
}

// TestNotificationDispatchRecordsExtractErrorMetric verifies that a failing
// notification extractor increments datalayer_extract_errors_total with the
// source and extractor type labels, mirroring the http polling path's recording contract.
func TestNotificationDispatchRecordsExtractErrorMetric(t *testing.T) {
	metrics.Reset()
	wantErr := errors.New("extract failed")
	gvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}
	failing := extractormocks.NewNotificationExtractor("failing").WithExtractError(wantErr)
	src := sourcenotifications.NewK8sNotificationSource(sourcenotifications.NotificationSourceType, "pods", gvk)
	reconciler := &notificationReconciler{
		src:        src,
		extractors: []fwkdl.NotificationExtractor{failing},
		log:        logr.Discard(),
	}
	event := &fwkdl.NotificationEvent{Object: &unstructured.Unstructured{}}

	if err := reconciler.dispatch(context.Background(), logr.Discard(), event); !errors.Is(err, wantErr) {
		t.Fatalf("dispatch() error = %v, want %v", err, wantErr)
	}

	if got := promtestutil.ToFloat64(
		metrics.LlmdDataLayerExtractErrorsTotal.WithLabelValues(src.TypedName().Type, failing.TypedName().Type)); got != 1.0 {
		t.Fatalf("datalayer_extract_errors_total = %v, want 1", got)
	}
}

// TestNotificationDispatchRecoversPanickingExtractor verifies that a panicking
// notification extractor is recovered, recorded in datalayer_extract_errors_total,
// and does not stop the dispatch loop.
func TestNotificationDispatchRecoversPanickingExtractor(t *testing.T) {
	metrics.Reset()
	gvk := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"}
	panicking := extractormocks.NewNotificationExtractor("panicking").WithExtractPanic("boom")
	working := extractormocks.NewNotificationExtractor("working")
	src := sourcenotifications.NewK8sNotificationSource(sourcenotifications.NotificationSourceType, "pods", gvk)
	reconciler := &notificationReconciler{
		src:        src,
		extractors: []fwkdl.NotificationExtractor{panicking, working},
		log:        logr.Discard(),
	}
	event := &fwkdl.NotificationEvent{Object: &unstructured.Unstructured{}}

	var panicked any
	var dispatchErr error
	func() {
		defer func() { panicked = recover() }()
		dispatchErr = reconciler.dispatch(context.Background(), logr.Discard(), event)
	}()
	if panicked != nil {
		t.Fatalf("dispatch panicked: %v", panicked)
	}
	if dispatchErr == nil {
		t.Fatalf("dispatch() = nil, want an error so the reconcile is retried after a panicking extractor")
	}

	if got := promtestutil.ToFloat64(
		metrics.LlmdDataLayerExtractErrorsTotal.WithLabelValues(src.TypedName().Type, panicking.TypedName().Type)); got != 1.0 {
		t.Fatalf("datalayer_extract_errors_total = %v, want 1", got)
	}
	if got := len(working.GetEvents()); got != 1 {
		t.Fatalf("working extractor received %d events, want 1", got)
	}
}

func TestHasNotificationExtractor(t *testing.T) {
	sliceGVK := schema.GroupVersionKind{Group: "discovery.k8s.io", Version: "v1", Kind: "EndpointSlice"}
	watcher := fwkplugin.TypedName{Type: "watcher", Name: "watcher"}
	skipped := fwkplugin.TypedName{Type: "skipped", Name: "skipped"}

	runtime := NewRuntime(time.Second)
	if err := runtime.Register(fwkdl.PendingRegistration{
		Owner:      watcher,
		SourceType: sourcenotifications.NotificationSourceType,
		Extractor:  extractormocks.NewNotificationExtractor("watcher-extractor").WithGVK(sliceGVK),
		DefaultSource: sourcenotifications.NewK8sNotificationSource(
			sourcenotifications.NotificationSourceType, "watcher/slices", sliceGVK),
	}); err != nil {
		t.Fatalf("Register: %v", err)
	}
	// No default source and none configured: skipped with a warning.
	if err := runtime.Register(fwkdl.PendingRegistration{
		Owner:      skipped,
		SourceType: "no-such-source",
		Extractor:  extractormocks.NewNotificationExtractor("skipped-extractor").WithGVK(sliceGVK),
		IfMissing:  fwkdl.Warn,
	}); err != nil {
		t.Fatalf("Register: %v", err)
	}
	if runtime.HasNotificationExtractor(watcher) {
		t.Fatal("HasNotificationExtractor() before Configure = true")
	}
	if err := runtime.Configure(&Config{}, logr.Discard()); err != nil {
		t.Fatalf("Configure: %v", err)
	}

	if !runtime.HasNotificationExtractor(watcher) {
		t.Error("HasNotificationExtractor(watcher) = false, want true")
	}
	if runtime.HasNotificationExtractor(skipped) {
		t.Error("HasNotificationExtractor(skipped) = true, want false")
	}
	if runtime.HasNotificationExtractor(fwkplugin.TypedName{Type: "other", Name: "other"}) {
		t.Error("HasNotificationExtractor(unregistered) = true, want false")
	}
}

func TestNotificationCacheByObject(t *testing.T) {
	podGVK := schema.GroupVersionKind{Version: "v1", Kind: "Pod"}
	sliceGVK := schema.GroupVersionKind{Group: "discovery.k8s.io", Version: "v1", Kind: "EndpointSlice"}
	cmGVK := schema.GroupVersionKind{Version: "v1", Kind: "ConfigMap"}
	sel := labels.SelectorFromSet(labels.Set{"kubernetes.io/service-name": "model-server"})

	runtime := NewRuntime(time.Second)
	newSource := func(name string, gvk schema.GroupVersionKind, opts ...sourcenotifications.K8sNotificationSourceOption) DataSourceConfig {
		return DataSourceConfig{
			Plugin:     sourcenotifications.NewK8sNotificationSource(sourcenotifications.NotificationSourceType, name, gvk, opts...),
			Extractors: []fwkplugin.Plugin{extractormocks.NewNotificationExtractor(name + "-extractor").WithGVK(gvk)},
		}
	}
	if err := runtime.Configure(&Config{Sources: []DataSourceConfig{
		newSource("pods", podGVK),
		newSource("slices", sliceGVK, sourcenotifications.WithNamespace("models"), sourcenotifications.WithLabelSelector(sel)),
		newSource("configmaps", cmGVK, sourcenotifications.WithLabelSelector(sel)),
	}}, logr.Discard()); err != nil {
		t.Fatalf("Configure: %v", err)
	}

	got := map[schema.GroupVersionKind]cache.ByObject{}
	for obj, byObject := range runtime.NotificationCacheByObject() {
		got[obj.GetObjectKind().GroupVersionKind()] = byObject
	}
	if len(got) != 2 {
		t.Fatalf("NotificationCacheByObject() has %d entries, want 2 (unscoped Pod source left out): %v", len(got), got)
	}
	slices, ok := got[sliceGVK]
	if !ok || len(slices.Namespaces) != 1 || slices.Namespaces["models"].LabelSelector.String() != sel.String() {
		t.Errorf("EndpointSlice scope = %+v, want namespace models with selector %s", slices, sel)
	}
	cms, ok := got[cmGVK]
	if !ok || cms.Namespaces != nil || cms.Label.String() != sel.String() {
		t.Errorf("ConfigMap scope = %+v, want all namespaces with selector %s", cms, sel)
	}
}
