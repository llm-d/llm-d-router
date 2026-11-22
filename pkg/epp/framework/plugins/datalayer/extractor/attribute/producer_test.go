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

package attribute

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	attrstring "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/string"
)

func TestFactory_Validation(t *testing.T) {
	cases := []struct {
		name    string
		params  string
		wantErr string
	}{
		{name: "empty attributeKeys rejected", params: `{"attributeKeys": []}`, wantErr: "non-empty"},
		{name: "empty entry rejected", params: `{"attributeKeys": ["pathPrefix", ""]}`, wantErr: "non-empty entries"},
		{name: "duplicate key rejected", params: `{"attributeKeys": ["a", "a"]}`, wantErr: "duplicate"},
		{name: "valid", params: `{"attributeKeys": ["pathPrefix", "model"]}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := Factory("test", json.NewDecoder(strings.NewReader(tc.params)), nil)
			if tc.wantErr != "" {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.wantErr)
				return
			}
			require.NoError(t, err)
		})
	}
}

func newEndpointWithAttributes(attrs map[string]string) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		ID:         types.NamespacedName{Name: "ep-1", Namespace: "default"},
		Attributes: attrs,
	}, nil)
}

func TestExtract_ProducesDynamicAttributes(t *testing.T) {
	p, err := Factory("test", json.NewDecoder(strings.NewReader(
		`{"attributeKeys": ["pathPrefix", "model"]}`)), nil)
	require.NoError(t, err)

	ep := newEndpointWithAttributes(map[string]string{"pathPrefix": "/laneA", "model": "fp8"})
	require.NoError(t, p.(fwkdl.EndpointExtractor).Extract(context.Background(),
		fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}))

	v, ok := attrstring.ReadValue(ep.GetAttributes(), fwkplugin.NewDataKey("pathPrefix", "test"))
	require.True(t, ok)
	assert.Equal(t, attrstring.Value("/laneA"), v)
	v, ok = attrstring.ReadValue(ep.GetAttributes(), fwkplugin.NewDataKey("model", "test"))
	require.True(t, ok)
	assert.Equal(t, attrstring.Value("fp8"), v)

	_, ok = attrstring.ReadValue(ep.GetAttributes(), fwkplugin.NewDataKey("missing", "test"))
	assert.False(t, ok, "unconfigured keys stay absent")
}

func TestProduces_DeclaresConfiguredKeys(t *testing.T) {
	p, err := Factory("test", json.NewDecoder(strings.NewReader(
		`{"attributeKeys": ["pathPrefix"]}`)), nil)
	require.NoError(t, err)
	produced := p.(fwkplugin.ProducerPlugin).Produces()
	_, ok := produced[fwkplugin.NewDataKey("pathPrefix", "test")]
	assert.True(t, ok, "configured key must be declared in Produces")
	assert.NotContains(t, produced, fwkplugin.NewDataKey("model", "test"))
}

// The attribute must resolve through the endpoint's current metadata on every
// read: an upsert that changes (or removes) the attributes is visible without
// re-stamping, unlike a write-once stamp at construction.
func TestExtract_ReadFollowsMetadataUpdates(t *testing.T) {
	p, err := Factory("test", json.NewDecoder(strings.NewReader(
		`{"attributeKeys": ["pathPrefix"]}`)), nil)
	require.NoError(t, err)

	ep := newEndpointWithAttributes(map[string]string{"pathPrefix": "/laneA"})
	extractor := p.(fwkdl.EndpointExtractor)
	require.NoError(t, extractor.Extract(context.Background(),
		fwkdl.EndpointEvent{Type: fwkdl.EventAddOrUpdate, Endpoint: ep}))

	key := fwkplugin.NewDataKey("pathPrefix", "test")
	v, ok := attrstring.ReadValue(ep.GetAttributes(), key)
	require.True(t, ok)
	assert.Equal(t, attrstring.Value("/laneA"), v)

	// Discovery reload changes the value: metadata is updated in place.
	ep.UpdateMetadata(&fwkdl.EndpointMetadata{
		ID:         types.NamespacedName{Name: "ep-1", Namespace: "default"},
		Attributes: map[string]string{"pathPrefix": "/laneB"},
	})
	v, ok = attrstring.ReadValue(ep.GetAttributes(), key)
	require.True(t, ok)
	assert.Equal(t, attrstring.Value("/laneB"), v, "read must follow the updated metadata")

	// Discovery reload removes the attribute entirely.
	ep.UpdateMetadata(&fwkdl.EndpointMetadata{
		ID: types.NamespacedName{Name: "ep-1", Namespace: "default"},
	})
	_, ok = attrstring.ReadValue(ep.GetAttributes(), key)
	assert.False(t, ok, "removed attribute must disappear from reads")
}
