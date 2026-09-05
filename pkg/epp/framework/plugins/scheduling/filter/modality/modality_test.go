package modality

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	"github.com/llm-d/llm-d-router/pkg/common"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	"github.com/llm-d/llm-d-router/test/utils"
)

func createEndpoint(name, ip string, labels map[string]string) scheduling.Endpoint {
	return scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			ID:      k8stypes.NamespacedName{Name: name},
			Address: ip,
			Labels:  labels,
		},
		&fwkdl.Metrics{},
		nil,
	)
}

func mixedEndpoints() []scheduling.Endpoint {
	return []scheduling.Endpoint{
		createEndpoint("tts-pod", "10.0.0.1",
			map[string]string{common.ModelArchLabel: common.ModelArchAutoRegressTTS}),
		createEndpoint("stt-pod", "10.0.0.2",
			map[string]string{common.ModelArchLabel: common.ModelArchEncoderDecSTT}),
		createEndpoint("diffusion-pod", "10.0.0.3",
			map[string]string{common.ModelArchLabel: common.ModelArchDiffusion}),
		createEndpoint("omni-pod", "10.0.0.4",
			map[string]string{common.ModelArchLabel: common.ModelArchOmniLLM}),
		createEndpoint("llm-pod", "10.0.0.5",
			map[string]string{common.ModelArchLabel: common.ModelArchAutoRegressLLM}),
		createEndpoint("no-label-pod", "10.0.0.6",
			map[string]string{"app": "vllm"}),
	}
}

func endpointNames(eps []scheduling.Endpoint) []string {
	names := make([]string, len(eps))
	for i, ep := range eps {
		names[i] = ep.GetMetadata().ID.Name
	}
	return names
}

func TestModalityFilter_AudioSpeech(t *testing.T) {
	ctx := utils.NewTestContext(t)
	f := NewModalityFilter()

	req := &scheduling.InferenceRequest{
		Headers: map[string]string{common.EnvoyPathHeader: "/v1/audio/speech"},
	}
	filtered := f.Filter(ctx, req, mixedEndpoints())

	assert.ElementsMatch(t, []string{"tts-pod", "omni-pod"}, endpointNames(filtered))
}

func TestModalityFilter_AudioTranscriptions(t *testing.T) {
	ctx := utils.NewTestContext(t)
	f := NewModalityFilter()

	req := &scheduling.InferenceRequest{
		Headers: map[string]string{common.EnvoyPathHeader: "/v1/audio/transcriptions"},
	}
	filtered := f.Filter(ctx, req, mixedEndpoints())

	assert.ElementsMatch(t, []string{"stt-pod"}, endpointNames(filtered))
}

func TestModalityFilter_ImagesGenerations(t *testing.T) {
	ctx := utils.NewTestContext(t)
	f := NewModalityFilter()

	req := &scheduling.InferenceRequest{
		Headers: map[string]string{common.EnvoyPathHeader: "/v1/images/generations"},
	}
	filtered := f.Filter(ctx, req, mixedEndpoints())

	assert.ElementsMatch(t, []string{"diffusion-pod"}, endpointNames(filtered))
}

func TestModalityFilter_UnknownPath(t *testing.T) {
	ctx := utils.NewTestContext(t)
	f := NewModalityFilter()

	req := &scheduling.InferenceRequest{
		Headers: map[string]string{common.EnvoyPathHeader: "/v1/chat/completions"},
	}
	all := mixedEndpoints()
	filtered := f.Filter(ctx, req, all)

	assert.Len(t, filtered, len(all))
}

func TestModalityFilter_QueryStringStripped(t *testing.T) {
	ctx := utils.NewTestContext(t)
	f := NewModalityFilter()

	req := &scheduling.InferenceRequest{
		Headers: map[string]string{common.EnvoyPathHeader: "/v1/audio/speech?model=tts-1"},
	}
	filtered := f.Filter(ctx, req, mixedEndpoints())

	assert.ElementsMatch(t, []string{"tts-pod", "omni-pod"}, endpointNames(filtered))
}

func TestModalityFilter_EmptyPath(t *testing.T) {
	ctx := utils.NewTestContext(t)
	f := NewModalityFilter()

	req := &scheduling.InferenceRequest{
		Headers: map[string]string{},
	}
	all := mixedEndpoints()
	filtered := f.Filter(ctx, req, all)

	assert.Len(t, filtered, len(all))
}

func TestModalityFilter_EmptyEndpoints(t *testing.T) {
	ctx := utils.NewTestContext(t)
	f := NewModalityFilter()

	req := &scheduling.InferenceRequest{
		Headers: map[string]string{common.EnvoyPathHeader: "/v1/audio/speech"},
	}
	filtered := f.Filter(ctx, req, []scheduling.Endpoint{})

	assert.Empty(t, filtered)
}

func TestModalityFilterFactory(t *testing.T) {
	p, err := ModalityFilterFactory("test-modality", nil, nil)
	require.NoError(t, err)

	mf, ok := p.(*ModalityFilter)
	require.True(t, ok)

	assert.Equal(t, ModalityFilterType, mf.TypedName().Type)
	assert.Equal(t, "test-modality", mf.TypedName().Name)
}
