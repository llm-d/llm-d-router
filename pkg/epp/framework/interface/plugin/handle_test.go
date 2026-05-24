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

package plugin

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"
)

// scorerPlugin is a typed plugin used to exercise PluginByType type assertions.
type scorerPlugin struct {
	basePlugin
}

// Score is unique to scorerPlugin so we can do a real interface-typed cast in tests.
type scorerInterface interface {
	Plugin
	Score() float64
}

func (s *scorerPlugin) Score() float64 { return 0.5 }

func TestNewEppHandle_ContextAndPodList(t *testing.T) {
	type ctxKey struct{}
	ctx := context.WithValue(context.Background(), ctxKey{}, "value")
	expectedPods := []types.NamespacedName{
		{Namespace: "ns1", Name: "pod-a"},
		{Namespace: "ns1", Name: "pod-b"},
	}

	h := NewEppHandle(ctx, func() []types.NamespacedName {
		return expectedPods
	})

	assert.Equal(t, "value", h.Context().Value(ctxKey{}))
	assert.Equal(t, expectedPods, h.PodList())
}

func TestEppHandle_PodListIsLazy(t *testing.T) {
	calls := 0
	h := NewEppHandle(context.Background(), func() []types.NamespacedName {
		calls++
		return nil
	})

	assert.Equal(t, 0, calls, "factory must not invoke PodList eagerly")
	h.PodList()
	h.PodList()
	assert.Equal(t, 2, calls)
}

func TestEppHandlePlugins_AddAndGet(t *testing.T) {
	h := NewEppHandle(context.Background(), func() []types.NamespacedName { return nil })

	p1 := &basePlugin{name: TypedName{Type: "filter", Name: "f1"}}
	p2 := &basePlugin{name: TypedName{Type: "scorer", Name: "s1"}}

	h.AddPlugin("f1", p1)
	h.AddPlugin("s1", p2)

	assert.Same(t, p1, h.Plugin("f1"))
	assert.Same(t, p2, h.Plugin("s1"))
	assert.Nil(t, h.Plugin("missing"))
}

func TestEppHandlePlugins_AddOverwrites(t *testing.T) {
	h := NewEppHandle(context.Background(), func() []types.NamespacedName { return nil })

	p1 := &basePlugin{name: TypedName{Type: "t", Name: "n"}}
	p2 := &basePlugin{name: TypedName{Type: "t", Name: "n"}}

	h.AddPlugin("shared", p1)
	h.AddPlugin("shared", p2)

	assert.Same(t, p2, h.Plugin("shared"))
}

func TestEppHandlePlugins_GetAllPlugins(t *testing.T) {
	h := NewEppHandle(context.Background(), func() []types.NamespacedName { return nil })

	all := h.GetAllPlugins()
	assert.Empty(t, all)

	p1 := &basePlugin{name: TypedName{Type: "a", Name: "p1"}}
	p2 := &basePlugin{name: TypedName{Type: "b", Name: "p2"}}
	h.AddPlugin("p1", p1)
	h.AddPlugin("p2", p2)

	all = h.GetAllPlugins()
	assert.Len(t, all, 2)
	assert.Contains(t, all, Plugin(p1))
	assert.Contains(t, all, Plugin(p2))
}

func TestEppHandlePlugins_GetAllPluginsWithNames(t *testing.T) {
	h := NewEppHandle(context.Background(), func() []types.NamespacedName { return nil })

	p := &basePlugin{name: TypedName{Type: "a", Name: "p1"}}
	h.AddPlugin("alias", p)

	named := h.GetAllPluginsWithNames()
	assert.Len(t, named, 1)
	assert.Same(t, p, named["alias"])
}

func TestPluginByType_Success(t *testing.T) {
	h := NewEppHandle(context.Background(), func() []types.NamespacedName { return nil })

	sp := &scorerPlugin{basePlugin: basePlugin{name: TypedName{Type: "scorer", Name: "kv"}}}
	h.AddPlugin("kv", sp)

	got, err := PluginByType[scorerInterface](h, "kv")
	assert.NoError(t, err)
	assert.Same(t, sp, got)
	assert.Equal(t, 0.5, got.Score())
}

func TestPluginByType_NotFound(t *testing.T) {
	h := NewEppHandle(context.Background(), func() []types.NamespacedName { return nil })

	got, err := PluginByType[scorerInterface](h, "absent")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "absent")
	assert.Nil(t, got)
}

func TestPluginByType_WrongType(t *testing.T) {
	h := NewEppHandle(context.Background(), func() []types.NamespacedName { return nil })

	// Register a plugin that does not satisfy scorerInterface.
	h.AddPlugin("bare", &basePlugin{name: TypedName{Type: "filter", Name: "bare"}})

	got, err := PluginByType[scorerInterface](h, "bare")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "bare")
	assert.Nil(t, got)
}
