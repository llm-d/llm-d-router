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

package predictedlatency

import (
	"bufio"
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"net/http/httputil"
	"net/url"
	"os"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/go-logr/logr"
	fwkrequest "github.com/llm-d/llm-d-router/pkg/epp/framework/common/request"
	fwkrh "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requesthandling"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	client "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requesthandling/dataproducer/predictedlatency/latencypredictorclient/protocol"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// This opt-in test drives the production hooks with real streaming inference.
// Endpoint metrics are fixed nonzero fixtures; token IDs and timing come from vLLM.
func TestLiveFeaturePipeline(t *testing.T) {
	backend := os.Getenv("LATENCY_TEST_VLLM_URL")
	if backend == "" {
		t.Skip("LATENCY_TEST_VLLM_URL is not set")
	}
	training := os.Getenv("TRAINING_SERVER_URL")
	prediction := os.Getenv("PREDICTION_SERVER_URL")
	require.NotEmpty(t, training)
	require.NotEmpty(t, prediction)
	target, err := url.Parse(training)
	require.NoError(t, err)
	proxy := httputil.NewSingleHostReverseProxy(target)
	var mu sync.Mutex
	var received []*protocol.TrainingEntry
	var decodeErrors []error
	tap := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		if request.Method == http.MethodPost && request.URL.Path == "/add_training_data_bulk" {
			data, readErr := io.ReadAll(request.Body)
			if readErr == nil {
				readErr = request.Body.Close()
			}
			request.Body = io.NopCloser(bytes.NewReader(data))
			batch := &protocol.BulkTrainingRequest{}
			if readErr == nil {
				readErr = protocol.Unmarshal(data, batch)
			}
			mu.Lock()
			if readErr != nil {
				decodeErrors = append(decodeErrors, readErr)
			} else {
				received = append(received, batch.GetEntries()...)
			}
			mu.Unlock()
		}
		proxy.ServeHTTP(writer, request)
	}))
	t.Cleanup(tap.Close)
	httpClient := &http.Client{Timeout: 30 * time.Second}
	modelDigest := func(name string) ([sha256.Size]byte, error) {
		var digest [sha256.Size]byte
		request, requestErr := http.NewRequestWithContext(t.Context(), http.MethodGet, training+"/model/"+name+"/download", nil)
		if requestErr != nil {
			return digest, requestErr
		}
		response, requestErr := httpClient.Do(request)
		if requestErr != nil {
			return digest, requestErr
		}
		defer response.Body.Close()
		if response.StatusCode != http.StatusOK {
			return digest, fmt.Errorf("model download returned %s", response.Status)
		}
		data, readErr := io.ReadAll(response.Body)
		return sha256.Sum256(data), readErr
	}
	initialTTFT, err := modelDigest("ttft")
	require.NoError(t, err)
	initialTPOT, err := modelDigest("tpot")
	require.NoError(t, err)
	requestJSON := func(path string, payload any, destination any) {
		t.Helper()
		method := http.MethodGet
		var body io.Reader
		if payload != nil {
			data, marshalErr := json.Marshal(payload)
			require.NoError(t, marshalErr)
			body = bytes.NewReader(data)
			method = http.MethodPost
		}
		request, requestErr := http.NewRequestWithContext(t.Context(), method, path, body)
		require.NoError(t, requestErr)
		request.Header.Set("Content-Type", "application/json")
		response, requestErr := httpClient.Do(request)
		require.NoError(t, requestErr)
		defer response.Body.Close()
		require.Equal(t, http.StatusOK, response.StatusCode)
		require.NoError(t, json.NewDecoder(response.Body).Decode(destination))
	}
	var models struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	requestJSON(backend+"/v1/models", nil, &models)
	require.NotEmpty(t, models.Data)
	model := models.Data[0].ID
	prompt := "The capital of France is"
	var tokens struct {
		Tokens []uint32 `json:"tokens"`
	}
	requestJSON(backend+"/tokenize", map[string]any{"model": model, "prompt": prompt}, &tokens)
	require.NotEmpty(t, tokens.Tokens)

	config := client.DefaultConfig()
	config.TrainingURL = tap.URL
	config.PredictionURLs = []string{prediction}
	config.FlushInterval = 100 * time.Millisecond
	predictor := client.New(config, logr.Discard())
	t.Cleanup(func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		predictor.Stop(ctx)
	})
	require.NoError(t, predictor.Start(t.Context()))
	producerConfig := DefaultConfig
	producerConfig.StreamingMode = true
	producer := NewPredictedLatency("live-feature-pipeline", producerConfig, predictor)
	t.Cleanup(producer.sloContextStore.Stop)
	endpoint := createTestEndpoint("live-vllm", 0.4, 3, 2)
	for index := range 3 {
		request := createTestInferenceRequest("live-"+strconv.Itoa(index), 10000, 10000)
		request.TargetModel = model
		request.Body.Completions.Prompt = fwkrequest.Prompt{Raw: prompt}
		request.Body.TokenizedRequest = &fwkrequest.TokenizedRequest{Prompts: []fwkrequest.PromptTokens{{TokenIDs: tokens.Tokens}}}
		require.NoError(t, producer.Produce(t.Context(), request, []fwksched.Endpoint{endpoint}))
		require.NoError(t, producer.PreRequest(t.Context(), request, &fwksched.SchedulingResult{
			PrimaryProfileName: "default", ProfileResults: map[string]*fwksched.ProfileRunResult{"default": {TargetEndpoints: []fwksched.Endpoint{endpoint}}},
		}))
		payload, marshalErr := json.Marshal(map[string]any{"model": model, "prompt": prompt, "max_tokens": 16, "temperature": 0, "stream": true})
		require.NoError(t, marshalErr)
		upstream, requestErr := http.NewRequestWithContext(t.Context(), http.MethodPost, backend+"/v1/completions", bytes.NewReader(payload))
		require.NoError(t, requestErr)
		upstream.Header.Set("Content-Type", "application/json")
		response, requestErr := httpClient.Do(upstream)
		require.NoError(t, requestErr)
		defer response.Body.Close()
		require.Equal(t, http.StatusOK, response.StatusCode)
		scanner := bufio.NewScanner(response.Body)
		finished := false
		chunks := 0
		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") || line == "data: [DONE]" {
				continue
			}
			var event struct {
				Choices []struct {
					Text         string  `json:"text"`
					FinishReason *string `json:"finish_reason"`
				} `json:"choices"`
			}
			require.NoError(t, json.Unmarshal([]byte(strings.TrimPrefix(line, "data: ")), &event))
			if len(event.Choices) == 0 {
				continue
			}
			choice := event.Choices[0]
			if choice.Text == "" && choice.FinishReason == nil {
				continue
			}
			chunks++
			finished = choice.FinishReason != nil
			producer.ResponseBody(t.Context(), request, &fwkrh.Response{EndOfStream: finished, TerminationCause: fwkrh.TerminationCauseNatural}, endpoint.GetMetadata())
		}
		require.NoError(t, scanner.Err())
		require.NoError(t, response.Body.Close())
		require.True(t, finished, "vLLM stream did not finish")
		require.Greater(t, chunks, 1)
	}
	require.Eventually(t, func() bool {
		mu.Lock()
		defer mu.Unlock()
		return len(received) >= 6
	}, 10*time.Second, 100*time.Millisecond)
	mu.Lock()
	entries := append([]*protocol.TrainingEntry(nil), received...)
	parseErrors := append([]error(nil), decodeErrors...)
	mu.Unlock()
	require.Empty(t, parseErrors)
	var ttft, tpot int
	for _, entry := range entries {
		features := entry.GetFeatures()
		assert.Equal(t, int64(len(tokens.Tokens)), features.GetInputTokenLength())
		assert.Equal(t, 0.4, features.GetKvCachePercentage())
		assert.Equal(t, int64(3), features.GetNumRequestRunning())
		assert.Equal(t, int64(2), features.GetNumRequestWaiting())
		assert.Zero(t, features.GetNumTokensGenerated())
		if entry.GetActualTtftMs() > 0 {
			ttft++
			assert.NotNil(t, entry.PredictedTtftMs)
		}
		if entry.GetActualTpotMs() > 0 {
			tpot++
			assert.NotNil(t, entry.PredictedTpotMs)
		}
	}
	assert.Equal(t, 3, ttft)
	assert.Equal(t, 3, tpot)
	var status struct {
		TrainingData struct {
			TTFT int `json:"ttft_samples"`
			TPOT int `json:"tpot_samples"`
		} `json:"training_data"`
		Ready bool `json:"model_ready"`
	}
	requestJSON(training+"/data/status", nil, &status)
	assert.True(t, status.Ready)
	assert.GreaterOrEqual(t, status.TrainingData.TTFT, ttft)
	assert.GreaterOrEqual(t, status.TrainingData.TPOT, tpot)
	// Readiness and retrain timestamps can also refer to the initial prior models.
	require.Eventually(t, func() bool {
		ttftDigest, ttftErr := modelDigest("ttft")
		tpotDigest, tpotErr := modelDigest("tpot")
		return ttftErr == nil && tpotErr == nil && ttftDigest != initialTTFT && tpotDigest != initialTPOT
	}, 30*time.Second, 250*time.Millisecond)
	var reloaded struct {
		Ready  bool `json:"is_ready"`
		Loaded bool `json:"loaded"`
	}
	requestJSON(prediction+"/reload", map[string]any{}, &reloaded)
	require.True(t, reloaded.Ready)
	require.True(t, reloaded.Loaded)
	next := client.PredictionFromFeatures(entries[0].GetFeatures())
	next.NumTokensGenerated = 1
	predicted, err := predictor.PredictBulkStrict(t.Context(), []client.PredictionRequest{next})
	require.NoError(t, err)
	require.Len(t, predicted.Predictions, 1)
	assert.GreaterOrEqual(t, predicted.Predictions[0].TTFT, float64(0))
	assert.GreaterOrEqual(t, predicted.Predictions[0].TPOT, float64(0))
	t.Logf("%d real streaming requests produced %d TTFT and %d TPOT entries accepted by Python", 3, ttft, tpot)
}
