// Package models
package models

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/epp/datalayer"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	extmodels "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/extractor/models"
)

func TestDatasource(t *testing.T) {
	srcPlugin, err := ModelDataSourceFactory("models-data-source",
		json.RawMessage(`{"scheme":"https","path":"/models","insecureSkipVerify":true}`), nil)
	assert.Nil(t, err, "failed to create http datasource")
	source := srcPlugin.(fwkdl.PollingDataSource)

	extPlugin, err := extmodels.ModelServerExtractorFactory("models-data-extractor", nil, nil)
	assert.Nil(t, err, "failed to create extractor")
	extractor := extPlugin.(fwkdl.Extractor)

	cfg := &datalayer.Config{
		Sources: []datalayer.DataSourceConfig{
			{
				Plugin:     source,
				Extractors: []fwkdl.ExtractorBase{extractor},
			},
		},
	}

	pollingInterval := 50 * time.Millisecond
	runtime := datalayer.NewRuntime(pollingInterval)

	err = runtime.Configure(cfg, true, "", logr.Logger{})
	assert.Nil(t, err, "failed to configure runtime")

	ctx := context.Background()
	pod := &fwkdl.EndpointMetadata{
		NamespacedName: types.NamespacedName{
			Name:      "pod1",
			Namespace: "default",
		},
		Address: "1.2.3.4:5678",
	}

	endpoint := runtime.NewEndpoint(ctx, pod, nil)
	assert.NotNil(t, endpoint, "failed to create endpoint")

	data, err := source.Poll(ctx, endpoint)
	assert.NotNil(t, err, "expected to fail to collect metrics")
	assert.Nil(t, data)
}
