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
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/workqueue"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	extractormocks "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/mocks"
	sourcenotifications "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
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
	initialSync.processed.Start(first)
	initialSync.processed.Start(second)

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

	initialSync.processed.Finished(first)
	if initialSync.hasSynced() {
		t.Fatal("initial sync completed before every queued key was processed")
	}

	initialSync.processed.Finished(second)
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
	if err := runtime.CheckReady(); err != nil {
		t.Fatalf("CheckReady() without notification sources: %v", err)
	}

	initialSync := newNotificationInitialSync("pods")
	runtime.notificationSyncs = []*notificationInitialSync{initialSync}
	if err := runtime.CheckReady(); err == nil || !strings.Contains(err.Error(), "pods") {
		t.Fatalf("CheckReady() error = %v, want pending source name", err)
	}

	initialSync.upstreamSynced.Store(true)
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
