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

package datalayer

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime/schema"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	extractormocks "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/mocks"
	sourcemocks "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/mocks"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/source/notifications"
)

// --- Register() validation ---

func TestRegister_NilExtractor(t *testing.T) {
	r := NewRuntime(0)
	err := r.Register(fwkdl.PendingRegistration{
		Owner:      fwkplugin.TypedName{Type: "test", Name: "test"},
		SourceType: notifications.EndpointNotificationSourceType,
		Extractor:  nil,
	})
	require.Error(t, err, "Register should reject nil Extractor")
}

func TestRegister_ValidRegistration(t *testing.T) {
	r := NewRuntime(0)
	ext := extractormocks.NewEndpointExtractor("ext1")
	err := r.Register(fwkdl.PendingRegistration{
		Owner:      fwkplugin.TypedName{Type: "test-plugin", Name: "test"},
		SourceType: notifications.EndpointNotificationSourceType,
		Extractor:  ext,
	})
	require.NoError(t, err)
	require.Len(t, r.pendingRegistrations, 1, "registration should be stored")
}

// --- Configure() with pending registrations ---

func TestConfigure_ExtractorWiredToUserConfiguredSource(t *testing.T) {
	// (a) extractor appended to matching user-configured source
	r := NewRuntime(0)
	logger := newTestLogger(t)

	src := notifications.NewEndpointDataSource(notifications.EndpointNotificationSourceType, "ep-src")
	ext := extractormocks.NewEndpointExtractor("ext-a")

	require.NoError(t, r.Register(fwkdl.PendingRegistration{
		Owner:      fwkplugin.TypedName{Type: "test-plugin", Name: "test"},
		SourceType: notifications.EndpointNotificationSourceType,
		Extractor:  ext,
	}))

	cfg := &Config{
		Sources: []DataSourceConfig{{Plugin: src}},
	}
	err := r.Configure(cfg, false, "", logger)
	require.NoError(t, err)

	exts, ok := r.extractors.Get("ep-src")
	require.True(t, ok, "extractors should have entry for ep-src")
	require.Len(t, exts, 1)
	assert.Equal(t, "ext-a", exts[0].TypedName().Name)
}

func TestConfigure_DefaultSourceAutoRegisteredWhenAbsent(t *testing.T) {
	// (b) DefaultSource auto-registered when source absent
	r := NewRuntime(0)
	logger := newTestLogger(t)

	defaultSrc := notifications.NewEndpointDataSource(notifications.EndpointNotificationSourceType, notifications.EndpointNotificationSourceType)
	ext := extractormocks.NewEndpointExtractor("ext-b")

	require.NoError(t, r.Register(fwkdl.PendingRegistration{
		Owner:         fwkplugin.TypedName{Type: "test-plugin", Name: "test"},
		SourceType:    notifications.EndpointNotificationSourceType,
		Extractor:     ext,
		DefaultSource: defaultSrc,
	}))

	// No user sources; default should be auto-created.
	err := r.Configure(nil, false, "", logger)
	require.NoError(t, err)

	// Source should be in the endpoint manager.
	src, ok := r.endpoint.Get(notifications.EndpointNotificationSourceType)
	require.True(t, ok, "auto-registered source should appear in endpoint manager")
	assert.Equal(t, notifications.EndpointNotificationSourceType, src.TypedName().Name)

	// Extractor should be wired.
	exts, ok := r.extractors.Get(notifications.EndpointNotificationSourceType)
	require.True(t, ok, "extractor should be wired to auto-registered source")
	require.Len(t, exts, 1)
}

func TestConfigure_FailPolicyMissingSource(t *testing.T) {
	// (c) Fail policy + nil DefaultSource + missing source → error returned
	r := NewRuntime(0)
	logger := newTestLogger(t)

	ext := extractormocks.NewEndpointExtractor("ext-c")
	require.NoError(t, r.Register(fwkdl.PendingRegistration{
		Owner:      fwkplugin.TypedName{Type: "test-plugin", Name: "test"},
		SourceType: notifications.EndpointNotificationSourceType,
		Extractor:  ext,
		IfMissing:  fwkdl.Fail,
	}))

	err := r.Configure(nil, false, "", logger)
	require.Error(t, err, "Fail policy should return error when source is absent")
}

func TestConfigure_WarnPolicyMissingSource(t *testing.T) {
	// (d) Warn policy + nil DefaultSource + missing source → continues without error
	r := NewRuntime(0)
	logger := newTestLogger(t)

	ext := extractormocks.NewEndpointExtractor("ext-d")
	require.NoError(t, r.Register(fwkdl.PendingRegistration{
		Owner:      fwkplugin.TypedName{Type: "test-plugin", Name: "test"},
		SourceType: notifications.EndpointNotificationSourceType,
		Extractor:  ext,
		IfMissing:  fwkdl.Warn,
	}))

	err := r.Configure(nil, false, "", logger)
	require.NoError(t, err, "Warn policy should not return error when source is absent")
}

func TestConfigure_DedupByExtractorType(t *testing.T) {
	// (g) dedup: extractor type already wired via config → code registration is a no-op
	r := NewRuntime(0)
	logger := newTestLogger(t)

	src := notifications.NewEndpointDataSource(notifications.EndpointNotificationSourceType, "ep-src")
	extFromConfig := extractormocks.NewEndpointExtractor("config-ext")
	extFromCode := extractormocks.NewEndpointExtractor("code-ext")

	// User config wires extFromConfig to src.
	cfg := &Config{
		Sources: []DataSourceConfig{{
			Plugin:     src,
			Extractors: []fwkdl.ExtractorBase{extFromConfig},
		}},
	}

	// Code registers extFromCode — same type as extFromConfig ("mock-endpoint-extractor").
	require.NoError(t, r.Register(fwkdl.PendingRegistration{
		Owner:      fwkplugin.TypedName{Type: "test-plugin", Name: "test"},
		SourceType: notifications.EndpointNotificationSourceType,
		Extractor:  extFromCode,
	}))

	err := r.Configure(cfg, false, "", logger)
	require.NoError(t, err)

	exts, ok := r.extractors.Get("ep-src")
	require.True(t, ok)
	require.Len(t, exts, 1, "code registration should be deduped; only config extractor present")
	assert.Equal(t, "config-ext", exts[0].TypedName().Name)
}

// Pins rejection of duplicate-Type extractors per source. Append dedups silently otherwise.
func TestConfigure_DuplicateExtractorTypePerSource(t *testing.T) {
	r := NewRuntime(0)
	logger := newTestLogger(t)

	src := notifications.NewEndpointDataSource(notifications.EndpointNotificationSourceType, "ep-src")
	ext1 := extractormocks.NewEndpointExtractor("ext-1")
	ext2 := extractormocks.NewEndpointExtractor("ext-2")

	cfg := &Config{
		Sources: []DataSourceConfig{{
			Plugin:     src,
			Extractors: []fwkdl.ExtractorBase{ext1, ext2},
		}},
	}

	err := r.Configure(cfg, false, "", logger)
	require.ErrorIs(t, err, ErrDuplicateExtractorType)
}

// TestConfigure_CrossVariantCollisionNoPending pins the behavior added by
// validateNoCrossVariantCollisions: two sources sharing a SourceType across
// different variants must fail Configure even when no pending extractor
// references the colliding type. findSourceByType-driven detection only
// fires on pending resolution; this fills the gap.
func TestConfigure_CrossVariantCollisionNoPending(t *testing.T) {
	const collidingType = "colliding-source-type"
	gvk := schema.GroupVersionKind{Group: "test.io", Version: "v1", Kind: "Probe"}

	cases := []struct {
		name    string
		sources []fwkdl.DataSource
	}{
		{
			name: "polling+notification",
			sources: []fwkdl.DataSource{
				sourcemocks.NewDataSource(fwkplugin.TypedName{Type: collidingType, Name: "polling-src"}),
				sourcemocks.NewNotificationSource(collidingType, "notif-src", gvk),
			},
		},
		{
			name: "polling+endpoint",
			sources: []fwkdl.DataSource{
				sourcemocks.NewDataSource(fwkplugin.TypedName{Type: collidingType, Name: "polling-src"}),
				notifications.NewEndpointDataSource(collidingType, "endpoint-src"),
			},
		},
		{
			name: "notification+endpoint",
			sources: []fwkdl.DataSource{
				sourcemocks.NewNotificationSource(collidingType, "notif-src", gvk),
				notifications.NewEndpointDataSource(collidingType, "endpoint-src"),
			},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := NewRuntime(0)
			srcCfgs := make([]DataSourceConfig, len(tc.sources))
			for i, s := range tc.sources {
				srcCfgs[i] = DataSourceConfig{Plugin: s}
			}
			err := r.Configure(&Config{Sources: srcCfgs}, false, "", newTestLogger(t))
			require.ErrorIs(t, err, ErrSourceTypeCollision)
		})
	}
}

func TestConfigure_CrossVariantSourceTypeCollisionRejected(t *testing.T) {
	const collidingType = "colliding-source-type"
	gvk := schema.GroupVersionKind{Group: "test.io", Version: "v1", Kind: "Probe"}

	polling := func() fwkdl.DataSource {
		return sourcemocks.NewDataSource(fwkplugin.TypedName{Type: collidingType, Name: "polling-src"})
	}
	notification := func() fwkdl.DataSource {
		return sourcemocks.NewNotificationSource(collidingType, "notif-src", gvk)
	}
	endpoint := func() fwkdl.DataSource {
		return notifications.NewEndpointDataSource(collidingType, "endpoint-src")
	}

	cases := []struct {
		name    string
		sources []fwkdl.DataSource
	}{
		{"polling+endpoint", []fwkdl.DataSource{polling(), endpoint()}},
		{"polling+notification", []fwkdl.DataSource{polling(), notification()}},
		{"notification+endpoint", []fwkdl.DataSource{notification(), endpoint()}},
		{"all three variants", []fwkdl.DataSource{polling(), notification(), endpoint()}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			r := NewRuntime(0)
			logger := newTestLogger(t)

			require.NoError(t, r.Register(fwkdl.PendingRegistration{
				Owner:      fwkplugin.TypedName{Type: "test-plugin", Name: "test"},
				SourceType: collidingType,
				Extractor:  extractormocks.NewEndpointExtractor("ext"),
			}))

			srcCfgs := make([]DataSourceConfig, len(tc.sources))
			for i, s := range tc.sources {
				srcCfgs[i] = DataSourceConfig{Plugin: s}
			}

			err := r.Configure(&Config{Sources: srcCfgs}, false, "", logger)
			require.ErrorIs(t, err, ErrSourceTypeCollision)
		})
	}
}
