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

package attributeweight

import (
	"context"
	"encoding/json"
	"math"
	"testing"

	"github.com/go-logr/logr/funcr"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrmetrics "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/metrics"
	attrstring "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/string"
	labelproducer "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/label"
)

const (
	testAttributeKey = "gpu.product"
	testProducer     = "endpoint-labels"
)

func pointer[T any](value T) *T { return &value }

func scorerParams() parameters {
	return parameters{
		AttributeKey: testAttributeKey,
		Producer:     pointer(testProducer),
		Weights:      map[string]float64{"H100": 4, "A100": 2, "L40S": 1},
	}
}

func TestFactory(t *testing.T) {
	plugin, err := Factory("gpu-weight", fwkplugin.StrictDecoder(json.RawMessage(`{
		"attributeKey":"gpu.product","producer":"endpoint-labels","weights":{"H100":4,"L40S":1}
	}`)), nil)
	require.NoError(t, err)
	assert.Equal(t, fwkplugin.TypedName{Type: EndpointAttributeWeightScorerType, Name: "gpu-weight"}, plugin.TypedName())
}

func TestValidation(t *testing.T) {
	for _, test := range []struct {
		name    string
		mutate  func(*parameters)
		wantErr string
	}{
		{name: "missing attribute", mutate: func(p *parameters) { p.AttributeKey = "" }, wantErr: "attributeKey"},
		{name: "missing producer", mutate: func(p *parameters) { p.Producer = nil }, wantErr: "producer"},
		{name: "missing weights", mutate: func(p *parameters) { p.Weights = nil }, wantErr: "weights"},
		{name: "zero weight", mutate: func(p *parameters) { p.Weights = map[string]float64{"H100": 0} }, wantErr: "positive"},
		{name: "non-finite weight", mutate: func(p *parameters) { p.Weights = map[string]float64{"H100": math.NaN()} }, wantErr: "finite"},
		{name: "normalization underflow", mutate: func(p *parameters) {
			p.Weights = map[string]float64{"largest": math.MaxFloat64, "smallest": math.SmallestNonzeroFloat64}
		}, wantErr: "underflows to zero"},
	} {
		t.Run(test.name, func(t *testing.T) {
			params := scorerParams()
			test.mutate(&params)
			scorer, err := NewEndpointAttributeWeightScorer("gpu-weight", params)
			require.Error(t, err)
			assert.Contains(t, err.Error(), test.wantErr)
			assert.Nil(t, scorer)
		})
	}
}

func TestConsumesCategoryAndScore(t *testing.T) {
	scorer, err := NewEndpointAttributeWeightScorer("gpu-weight", scorerParams())
	require.NoError(t, err)
	key := fwkplugin.NewDataKey(testAttributeKey, testProducer)
	assert.IsType(t, attrstring.Value(""), scorer.Consumes().Optional[key])
	assert.Empty(t, scorer.Consumes().Required)
	assert.Equal(t, fwksched.Affinity, scorer.Category())

	newEndpoint := func(name string, value *string) fwksched.Endpoint {
		attrs := fwkdl.NewAttributes()
		if value != nil {
			attrs.Put(key, attrstring.Value(*value))
		}
		return fwksched.NewEndpoint(&fwkdl.EndpointMetadata{ID: types.NamespacedName{Name: name}}, nil, attrs)
	}

	h100, a100, unknown := "H100", "A100", "unknown"
	otherProducer := newEndpoint("other-producer", nil)
	otherProducer.Put(fwkplugin.NewDataKey(testAttributeKey, "other-labels"), attrstring.Value(h100))
	wrongType := newEndpoint("wrong-type", nil)
	wrongType.Put(key, attrmetrics.ScalarMetricValue(1))
	endpoints := []fwksched.Endpoint{
		newEndpoint("h100", &h100),
		newEndpoint("a100", &a100),
		newEndpoint("unknown", &unknown),
		newEndpoint("missing", nil),
		otherProducer,
		wrongType,
	}
	scores := scorer.Score(context.Background(), &fwksched.InferenceRequest{}, endpoints)

	assert.Equal(t, 1.0, scores[endpoints[0]])
	assert.Equal(t, 0.5, scores[endpoints[1]])
	assert.Equal(t, 0.25, scores[endpoints[2]])
	assert.Equal(t, 0.25, scores[endpoints[3]])
	assert.Equal(t, 0.25, scores[otherProducer])
	assert.Equal(t, 0.25, scores[wrongType])
}

func TestLabelProducerDependencies(t *testing.T) {
	for _, test := range []struct {
		name           string
		producerName   string
		attributeKey   string
		scorerProducer string
		omitProducer   bool
		wantWarning    bool
	}{
		{name: "default producer", producerName: labelproducer.LabelProducerType, attributeKey: testAttributeKey},
		{name: "named producer", producerName: "z-labels", attributeKey: testAttributeKey},
		{name: "missing producer", producerName: "z-labels", omitProducer: true, wantWarning: true},
		{name: "wrong producer name", producerName: "other-labels", attributeKey: testAttributeKey, scorerProducer: "z-labels", wantWarning: true},
		{name: "wrong attribute key", producerName: "z-labels", attributeKey: "region", wantWarning: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			var logs string
			logger := funcr.New(func(_, args string) { logs += args }, funcr.Options{})
			ctx := log.IntoContext(context.Background(), logger)
			handle := fwkplugin.NewEppHandle(ctx, nil)
			endpoint := fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{Labels: map[string]string{"nvidia.com/gpu.product": "H100"}}, nil)
			params := scorerParams()
			params.Producer = pointer(test.producerName)
			if test.scorerProducer != "" {
				params.Producer = pointer(test.scorerProducer)
			}
			scorer, err := NewEndpointAttributeWeightScorer("a-weight", params)
			require.NoError(t, err)
			handle.AddPlugin(scorer.TypedName().Name, scorer)
			if !test.omitProducer {
				producer, err := labelproducer.Factory(test.producerName, fwkplugin.StrictDecoder(json.RawMessage(`{
					"labels":[{"label":"nvidia.com/gpu.product","attributeKey":"`+test.attributeKey+`"}]
				}`)), handle)
				require.NoError(t, err)
				handle.AddPlugin(producer.TypedName().Name, producer)
				require.NoError(t, producer.(fwkdl.EndpointExtractor).Extract(ctx, fwkdl.EndpointEvent{
					Type: fwkdl.EventAddOrUpdate, Endpoint: endpoint,
				}))
			}

			require.NoError(t, datalayer.CreateMissingDataProducers(ctx, nil, nil, handle))
			scheduledEndpoint := fwksched.NewEndpoint(endpoint.GetMetadata(), nil, endpoint.GetAttributes())
			scores := scorer.Score(ctx, nil, []fwksched.Endpoint{scheduledEndpoint})
			if test.wantWarning {
				assert.Contains(t, logs, "Warning: optional data key has no producer")
				assert.Contains(t, logs, scorer.dataKey.String())
				assert.Contains(t, logs, scorer.TypedName().Name)
				assert.Equal(t, 0.25, scores[scheduledEndpoint])
				return
			}
			assert.Empty(t, logs)
			assert.Equal(t, 1.0, scores[scheduledEndpoint])

			producer := handle.Plugin(test.producerName)
			for _, plugins := range [][]fwkplugin.Plugin{
				{scorer, producer},
				{producer, scorer},
			} {
				ordered, err := datalayer.ValidateAndOrderDataDependencies(plugins)
				require.NoError(t, err)
				assert.Equal(t, []string{producer.TypedName().String(), scorer.TypedName().String()}, ordered)
			}
		})
	}
}
