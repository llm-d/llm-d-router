package modality

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
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
			map[string]string{ModelArchLabel: ModelArchAutoRegressTTS}),
		createEndpoint("stt-pod", "10.0.0.2",
			map[string]string{ModelArchLabel: ModelArchEncoderDecSTT}),
		createEndpoint("diffusion-pod", "10.0.0.3",
			map[string]string{ModelArchLabel: ModelArchDiffusion}),
		createEndpoint("omni-pod", "10.0.0.4",
			map[string]string{ModelArchLabel: ModelArchOmniLLM}),
		createEndpoint("llm-pod", "10.0.0.5",
			map[string]string{ModelArchLabel: ModelArchAutoRegressLLM}),
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
		Body: &fwkrh.InferenceRequestBody{TextToSpeech: &fwkrh.TextToSpeechRequest{Input: "hello"}},
	}
	filtered := f.Filter(ctx, req, mixedEndpoints())

	assert.ElementsMatch(t, []string{"tts-pod", "omni-pod"}, endpointNames(filtered))
}

func TestModalityFilter_AudioTranscriptions(t *testing.T) {
	ctx := utils.NewTestContext(t)
	f := NewModalityFilter()

	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{Transcriptions: &fwkrh.TranscriptionsRequest{Language: "en"}},
	}
	filtered := f.Filter(ctx, req, mixedEndpoints())

	assert.ElementsMatch(t, []string{"stt-pod"}, endpointNames(filtered))
}

func TestModalityFilter_ImagesGenerations(t *testing.T) {
	ctx := utils.NewTestContext(t)
	f := NewModalityFilter()

	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{Images: &fwkrh.ImagesGenerationsRequest{Prompt: "a dog"}},
	}
	filtered := f.Filter(ctx, req, mixedEndpoints())

	assert.ElementsMatch(t, []string{"diffusion-pod"}, endpointNames(filtered))
}

func TestModalityFilter_UnconstrainedRequestType(t *testing.T) {
	ctx := utils.NewTestContext(t)
	f := NewModalityFilter()

	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{ChatCompletions: &fwkrh.ChatCompletionsRequest{}},
	}
	all := mixedEndpoints()
	filtered := f.Filter(ctx, req, all)

	assert.Len(t, filtered, len(all))
}

func TestModalityFilter_NilBody(t *testing.T) {
	ctx := utils.NewTestContext(t)
	f := NewModalityFilter()

	req := &scheduling.InferenceRequest{}
	all := mixedEndpoints()
	filtered := f.Filter(ctx, req, all)

	assert.Len(t, filtered, len(all))
}

func TestModalityFilter_EmptyEndpoints(t *testing.T) {
	ctx := utils.NewTestContext(t)
	f := NewModalityFilter()

	req := &scheduling.InferenceRequest{
		Body: &fwkrh.InferenceRequestBody{TextToSpeech: &fwkrh.TextToSpeechRequest{Input: "hello"}},
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
