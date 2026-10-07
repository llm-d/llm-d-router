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

package proxy

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"maps"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/go-logr/logr"
	"github.com/stretchr/testify/require"

	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
	fwkrc "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/requestcontrol"
	sessionutil "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/scheduling/util/sessionaffinity"
	"github.com/llm-d/llm-d-router/pkg/sidecar/constants"
)

const (
	testPodName      = "decode-0"
	testPodNamespace = "default"
)

func newBidirectionalTestServer(t *testing.T, dataParallelSize int) *Server {
	t.Helper()
	s := NewProxy(Config{
		KVConnector:                constants.KVConnectorNIXLV2,
		BidirectionalKVXfer:        true,
		BidirectionalSessionHeader: sessionutil.DefaultHeader,
		BidirectionalCacheSize:     16,
		BidirectionalCacheTTL:      time.Minute,
		PodName:                    testPodName,
		PodNamespace:               testPodNamespace,
		DataParallelSize:           dataParallelSize,
	})
	s.logger = logr.Discard()
	return s
}

// eppSessionToken returns the session token the EPP's encoded-endpoint
// session affinity writes for the given rank endpoint of a pod.
func eppSessionToken(namespace, pod string, rank int) string {
	resp := &fwkrc.Response{}
	endpoint := &fwkdl.EndpointMetadata{ID: fwkdl.ID{Namespace: namespace, Name: routing.EndpointName(pod, rank)}}
	sessionutil.WriteResponseHeader(context.Background(), "test", sessionutil.DefaultHeader, resp, endpoint)
	return resp.Headers[sessionutil.DefaultHeader]
}

func requestWithSession(token string) *http.Request {
	r := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, nil)
	if token != "" {
		r.Header.Set(sessionutil.DefaultHeader, token)
	}
	return r
}

func chatMessage(role, content string) map[string]any {
	return map[string]any{reqcommon.FieldRole: role, reqcommon.FieldContent: content}
}

// requestBody returns a chat request body in the shape the proxy hands to the
// connector: only the fields decodeRequestBody inspects are decoded, and the rest,
// messages included, stay json.RawMessage. extra overrides or adds top-level fields.
func requestBody(extra map[string]any, messages ...any) map[string]any {
	fields := map[string]any{reqcommon.FieldModel: "m", reqcommon.FieldMessages: messages}
	maps.Copy(fields, extra)
	raw, err := json.Marshal(fields)
	if err != nil {
		panic(err)
	}
	body, err := decodeRequestBody(raw)
	if err != nil {
		panic(err)
	}
	return body
}

func chatBody(messages ...any) map[string]any { return requestBody(nil, messages...) }

// nixlDecodeResponse is a non-streaming decode response whose first choice is
// message and whose kv_transfer_params carry the decode engine's blocks.
func nixlDecodeResponse(t *testing.T, message map[string]any) []byte {
	t.Helper()
	b, err := json.Marshal(map[string]any{
		"id":     "chatcmpl-1",
		"object": "chat.completion",
		"choices": []any{map[string]any{
			"index": 0, "message": message, "finish_reason": "stop",
		}},
		"kv_transfer_params": map[string]any{
			"do_remote_prefill":         false,
			"do_remote_decode":          true,
			"remote_block_ids":          []any{[]any{1, 2, 3}},
			"remote_engine_id":          "decode-engine",
			"remote_request_id":         "chatcmpl-1",
			"remote_host":               "10.0.0.7",
			"remote_port":               5600,
			"remote_num_tokens":         192,
			"remote_blocks_expiry_time": 123.5,
			"tp_size":                   1,
			"dcp_size":                  1,
			"pp_size":                   1,
			"transfer_mode":             "read",
			"not_part_of_the_contract":  "dropped",
		},
	})
	require.NoError(t, err)
	return b
}

func captureJSON(t *testing.T, body []byte) *decodeCapture {
	t.Helper()
	rec := httptest.NewRecorder()
	rec.Header().Set("Content-Type", "application/json")
	w, c := newDecodeCapture(rec)
	_, err := w.Write(body)
	require.NoError(t, err)
	require.JSONEq(t, string(body), rec.Body.String(), "the capture must forward the response unchanged")
	return c
}

func TestSessionTargetsThisPod(t *testing.T) {
	tests := []struct {
		name  string
		dp    int
		token string
		want  bool
	}{
		{"EPP token for rank 0", 1, eppSessionToken(testPodNamespace, testPodName, 0), true},
		{"EPP token for rank 1 of 2", 2, eppSessionToken(testPodNamespace, testPodName, 1), true},
		{"EPP token for a rank beyond the data-parallel size", 2, eppSessionToken(testPodNamespace, testPodName, 2), false},
		{"EPP token for another pod", 1, eppSessionToken(testPodNamespace, "decode-1", 0), false},
		{"EPP token for another namespace", 1, eppSessionToken("other", testPodName, 0), false},
		{"pod name without the EPP rank suffix", 1, base64.StdEncoding.EncodeToString([]byte(testPodNamespace + "/" + testPodName)), false},
		{"not base64", 1, "!!!", false},
		{"no header", 1, "", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := newBidirectionalTestServer(t, tc.dp)
			require.Equal(t, tc.want, s.sessionTargetsThisPod(requestWithSession(tc.token)))
		})
	}

	t.Run("a server that does not know its own identity trusts no token", func(t *testing.T) {
		s := newBidirectionalTestServer(t, 1)
		s.config.PodName = ""
		require.False(t, s.sessionTargetsThisPod(requestWithSession(eppSessionToken(testPodNamespace, testPodName, 0))))
	})
}

func TestNewKVReuseGating(t *testing.T) {
	token := eppSessionToken(testPodNamespace, testPodName, 0)
	body := func() map[string]any { return chatBody(chatMessage("user", "hello")) }

	t.Run("applies to a chat request, routed when the token names this pod", func(t *testing.T) {
		s := newBidirectionalTestServer(t, 1)
		reuse := s.newKVReuse(requestWithSession(token), body(), reqcommon.APITypeChatCompletions)
		require.NotNil(t, reuse)
		require.True(t, reuse.routed)
	})
	t.Run("applies to the first turn, which carries no token yet", func(t *testing.T) {
		s := newBidirectionalTestServer(t, 1)
		reuse := s.newKVReuse(requestWithSession(""), body(), reqcommon.APITypeChatCompletions)
		require.NotNil(t, reuse)
		require.False(t, reuse.routed)
	})
	t.Run("a token for another pod is not routed", func(t *testing.T) {
		s := newBidirectionalTestServer(t, 1)
		reuse := s.newKVReuse(requestWithSession(eppSessionToken(testPodNamespace, "decode-1", 0)), body(), reqcommon.APITypeChatCompletions)
		require.NotNil(t, reuse)
		require.False(t, reuse.routed)
	})
	t.Run("not when the feature is off", func(t *testing.T) {
		s := newBidirectionalTestServer(t, 1)
		s.config.BidirectionalKVXfer = false
		s.kvReuseCache = nil
		require.Nil(t, s.newKVReuse(requestWithSession(token), body(), reqcommon.APITypeChatCompletions))
	})
	t.Run("not for another API", func(t *testing.T) {
		s := newBidirectionalTestServer(t, 1)
		require.Nil(t, s.newKVReuse(requestWithSession(token), body(), reqcommon.APITypeCompletions))
	})
	t.Run("not for a multi-choice request", func(t *testing.T) {
		s := newBidirectionalTestServer(t, 1)
		b := requestBody(map[string]any{"n": 2}, chatMessage("user", "hello"))
		require.Nil(t, s.newKVReuse(requestWithSession(token), b, reqcommon.APITypeChatCompletions))
	})
	t.Run("a single-choice request may say so explicitly", func(t *testing.T) {
		s := newBidirectionalTestServer(t, 1)
		for _, n := range []any{1, nil} {
			b := requestBody(map[string]any{"n": n}, chatMessage("user", "hello"))
			require.NotNil(t, s.newKVReuse(requestWithSession(token), b, reqcommon.APITypeChatCompletions), "n=%v", n)
		}
	})
	t.Run("not for a request that truncates the prompt", func(t *testing.T) {
		s := newBidirectionalTestServer(t, 1)
		b := requestBody(map[string]any{"truncate_prompt_tokens": 512}, chatMessage("user", "hello"))
		require.Nil(t, s.newKVReuse(requestWithSession(token), b, reqcommon.APITypeChatCompletions))
	})
	t.Run("not without messages", func(t *testing.T) {
		s := newBidirectionalTestServer(t, 1)
		b, err := decodeRequestBody([]byte(`{"model":"m"}`))
		require.NoError(t, err)
		require.Nil(t, s.newKVReuse(requestWithSession(token), b, reqcommon.APITypeChatCompletions))
	})
	t.Run("not in MoRI-IO write mode", func(t *testing.T) {
		s := newBidirectionalTestServer(t, 1)
		s.config.MoRIIOWriteMode = true
		require.Nil(t, s.newKVReuse(requestWithSession(token), body(), reqcommon.APITypeChatCompletions))
	})
}

// TestKVReuseReplaysOnlyAnExtendedHistory pins the isolation contract: cached
// blocks reach only a request whose messages extend the turn that produced
// them, because the engine copies them positionally without comparing tokens.
func TestKVReuseReplaysOnlyAnExtendedHistory(t *testing.T) {
	tools := []any{map[string]any{"type": "function", "function": map[string]any{"name": "lookup"}}}
	echoWithExtraFields := map[string]any{
		reqcommon.FieldRole: "assistant", reqcommon.FieldContent: "hi there",
		"refusal": nil, "annotations": []any{},
	}
	msgs := func() []any {
		return []any{chatMessage("user", "hello"), chatMessage("assistant", "hi there"), chatMessage("user", "more")}
	}
	with := func(extra map[string]any) map[string]any { return requestBody(extra, msgs()...) }

	tests := []struct {
		name     string
		followUp map[string]any
		wantHit  bool
	}{
		{"extends the turn", chatBody(chatMessage("user", "hello"), chatMessage("assistant", "hi there"), chatMessage("user", "more")), true},
		{"the echoed reply carries fields the decoder did not send", chatBody(chatMessage("user", "hello"), echoWithExtraFields, chatMessage("user", "more")), true},
		{"a different tenant (cache_salt)", with(map[string]any{"cache_salt": "tenant-b"}), false},
		{"a different model", with(map[string]any{"model": "other"}), false},
		{"a different tool set", with(map[string]any{"tools": tools}), false},
		{"a different reasoning effort", with(map[string]any{"reasoning_effort": "high"}), false},
		{"a different tool_choice", with(map[string]any{"tool_choice": "none"}), false},
		{"a different add_special_tokens", with(map[string]any{"add_special_tokens": false}), false},
		{"an edited earlier message", chatBody(chatMessage("user", "hola"), chatMessage("assistant", "hi there"), chatMessage("user", "more")), false},
		{"an edited reply", chatBody(chatMessage("user", "hello"), chatMessage("assistant", "hello there"), chatMessage("user", "more")), false},
		{"an unrelated conversation", chatBody(chatMessage("user", "other"), chatMessage("assistant", "hi there"), chatMessage("user", "more")), false},
		{"no new message after the reply", chatBody(chatMessage("user", "hello"), chatMessage("assistant", "hi there")), false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := newBidirectionalTestServer(t, 1)
			token := eppSessionToken(testPodNamespace, testPodName, 0)

			// The first turn precedes the response that issues the session token.
			first := s.newKVReuse(requestWithSession(""), chatBody(chatMessage("user", "hello")), reqcommon.APITypeChatCompletions)
			require.NotNil(t, first)
			require.Nil(t, first.take(), "a cold cache has nothing to replay")
			require.True(t, first.store(captureJSON(t, nixlDecodeResponse(t, chatMessage("assistant", "hi there")))))

			second := s.newKVReuse(requestWithSession(token), tc.followUp, reqcommon.APITypeChatCompletions)
			require.NotNil(t, second)
			got := second.take()
			if !tc.wantHit {
				require.Nil(t, got)
				return
			}
			require.NotNil(t, got)
			require.Equal(t, "decode-engine", got[reqcommon.FieldRemoteEngineID])
		})
	}
}

func TestKVReuseEntryIsSingleUse(t *testing.T) {
	s := newBidirectionalTestServer(t, 1)
	token := eppSessionToken(testPodNamespace, testPodName, 0)
	first := s.newKVReuse(requestWithSession(token), chatBody(chatMessage("user", "hello")), reqcommon.APITypeChatCompletions)
	require.True(t, first.store(captureJSON(t, nixlDecodeResponse(t, chatMessage("assistant", "hi there")))))

	followUp := func() map[string]any {
		return chatBody(chatMessage("user", "hello"), chatMessage("assistant", "hi there"), chatMessage("user", "more"))
	}
	require.NotNil(t, s.newKVReuse(requestWithSession(token), followUp(), reqcommon.APITypeChatCompletions).take())
	require.Nil(t, s.newKVReuse(requestWithSession(token), followUp(), reqcommon.APITypeChatCompletions).take(),
		"the prefill read releases the decode-side blocks, so a second replay would name freed blocks")
}

func TestKVReusePrefersTheLongestHistory(t *testing.T) {
	s := newBidirectionalTestServer(t, 1)
	token := eppSessionToken(testPodNamespace, testPodName, 0)

	turn1 := chatBody(chatMessage("user", "hello"))
	require.True(t, s.newKVReuse(requestWithSession(token), turn1, reqcommon.APITypeChatCompletions).
		store(captureJSON(t, nixlDecodeResponse(t, chatMessage("assistant", "one")))))

	turn2 := chatBody(chatMessage("user", "hello"), chatMessage("assistant", "one"), chatMessage("user", "next"))
	reply2 := nixlDecodeResponse(t, chatMessage("assistant", "two"))
	var parsed map[string]any
	require.NoError(t, json.Unmarshal(reply2, &parsed))
	parsed["kv_transfer_params"].(map[string]any)["remote_engine_id"] = "turn-2-engine"
	reply2, err := json.Marshal(parsed)
	require.NoError(t, err)
	require.True(t, s.newKVReuse(requestWithSession(token), turn2, reqcommon.APITypeChatCompletions).store(captureJSON(t, reply2)))

	turn3 := chatBody(chatMessage("user", "hello"), chatMessage("assistant", "one"), chatMessage("user", "next"),
		chatMessage("assistant", "two"), chatMessage("user", "last"))
	got := s.newKVReuse(requestWithSession(token), turn3, reqcommon.APITypeChatCompletions).take()
	require.NotNil(t, got)
	require.Equal(t, "turn-2-engine", got[reqcommon.FieldRemoteEngineID])
}

func TestKVReuseHandlesToolCallTurns(t *testing.T) {
	s := newBidirectionalTestServer(t, 1)
	token := eppSessionToken(testPodNamespace, testPodName, 0)
	toolCall := map[string]any{
		"id": "call_1", "type": "function",
		"function": map[string]any{"name": "lookup", "arguments": `{"q":1}`},
	}

	first := s.newKVReuse(requestWithSession(token), chatBody(chatMessage("user", "find it")), reqcommon.APITypeChatCompletions)
	reply := map[string]any{reqcommon.FieldRole: "assistant", reqcommon.FieldContent: nil, "tool_calls": []any{toolCall}}
	require.True(t, first.store(captureJSON(t, nixlDecodeResponse(t, reply))))

	toolResult := map[string]any{reqcommon.FieldRole: "tool", "tool_call_id": "call_1", reqcommon.FieldContent: "found"}
	followUp := func(call map[string]any) map[string]any {
		echo := map[string]any{reqcommon.FieldRole: "assistant", reqcommon.FieldContent: "", "tool_calls": []any{call}}
		return chatBody(chatMessage("user", "find it"), echo, toolResult)
	}

	// Different arguments are a different history. Checked first so the entry
	// is still there to be wrongly matched.
	other := map[string]any{"id": "call_1", "type": "function", "function": map[string]any{"name": "lookup", "arguments": `{"q":2}`}}
	require.Nil(t, s.newKVReuse(requestWithSession(token), followUp(other), reqcommon.APITypeChatCompletions).take())

	got := s.newKVReuse(requestWithSession(token), followUp(toolCall), reqcommon.APITypeChatCompletions).take()
	require.NotNil(t, got, "an agentic follow-up ends in a tool result, not a user message")
}

func TestKVReuseStoreRejectsIncompleteDecodeResponses(t *testing.T) {
	s := newBidirectionalTestServer(t, 1)
	token := eppSessionToken(testPodNamespace, testPodName, 0)

	tests := map[string]func(params map[string]any){
		"missing remote_request_id": func(p map[string]any) { delete(p, "remote_request_id") },
		"missing remote_engine_id":  func(p map[string]any) { delete(p, "remote_engine_id") },
		"missing remote_host":       func(p map[string]any) { delete(p, "remote_host") },
		"missing remote_port":       func(p map[string]any) { delete(p, "remote_port") },
		"no remote blocks":          func(p map[string]any) { p["remote_block_ids"] = []any{} },
		"not a decode-side response": func(p map[string]any) {
			p["do_remote_decode"] = false
		},
	}
	for name, mutate := range tests {
		t.Run(name, func(t *testing.T) {
			var parsed map[string]any
			require.NoError(t, json.Unmarshal(nixlDecodeResponse(t, chatMessage("assistant", "hi")), &parsed))
			mutate(parsed["kv_transfer_params"].(map[string]any))
			body, err := json.Marshal(parsed)
			require.NoError(t, err)

			reuse := s.newKVReuse(requestWithSession(token), chatBody(chatMessage("user", "hello")), reqcommon.APITypeChatCompletions)
			require.False(t, reuse.store(captureJSON(t, body)))
		})
	}
}

func TestInjectBidirectionalKVParamsCopiesOnlyTheNIXLContract(t *testing.T) {
	cached, ok := completeKVParams(map[string]any{
		"do_remote_prefill":         false,
		"do_remote_decode":          true,
		"remote_block_ids":          []any{[]any{1}},
		"remote_engine_id":          "e",
		"remote_request_id":         "r",
		"remote_host":               "h",
		"remote_port":               json.Number("5600"),
		"remote_num_tokens":         json.Number("64"),
		"remote_blocks_expiry_time": json.Number("1.5"),
		"tp_size":                   json.Number("1"),
		"dcp_size":                  json.Number("1"),
		"pp_size":                   json.Number("1"),
		"transfer_mode":             "read",
		"remote_kv_source":          map[string]any{"remote_host": "elsewhere"},
		"extra":                     "x",
	})
	require.True(t, ok)

	prefill := map[string]any{
		"do_remote_decode":  true,
		"do_remote_prefill": false,
		"remote_engine_id":  nil,
		"remote_block_ids":  nil,
		"remote_host":       nil,
		"remote_port":       nil,
	}
	injectBidirectionalKVParams(prefill, cached)

	require.Equal(t, true, prefill["do_remote_decode"])
	require.Equal(t, false, prefill["do_remote_prefill"], "the sidecar owns the prefill leg's role flags")
	for _, field := range []string{"remote_engine_id", "remote_request_id", "remote_host", "remote_port", "remote_block_ids",
		"remote_num_tokens", "remote_blocks_expiry_time", "tp_size", "dcp_size", "pp_size", "transfer_mode"} {
		require.Contains(t, prefill, field)
		require.NotNil(t, prefill[field], field)
	}
	require.NotContains(t, prefill, "extra")
	require.NotContains(t, prefill, "remote_kv_source", "the P2P source is composed by the sidecar, not replayed")
}

func TestKVReuseCacheEntriesExpire(t *testing.T) {
	cache := newKVReuseCache(Config{BidirectionalKVXfer: true, BidirectionalCacheSize: 4, BidirectionalCacheTTL: 150 * time.Millisecond})
	cache.Add("k", &kvReuseEntry{params: map[string]any{"remote_engine_id": "e"}})
	_, ok := cache.Get("k")
	require.True(t, ok)
	time.Sleep(350 * time.Millisecond)
	_, ok = cache.Get("k")
	require.False(t, ok, "an entry outlives neither its TTL nor the engine's blocks")

	require.Nil(t, newKVReuseCache(Config{}), "no cache when the feature is off")
}

func TestCloneSharesTheKVReuseCacheAcrossRanks(t *testing.T) {
	s := newBidirectionalTestServer(t, 2)
	rankOne := s.Clone()
	rankOne.logger = logr.Discard()
	require.Same(t, s.kvReuseCache, rankOne.kvReuseCache)

	// A turn served on rank 0 is replayed on a follow-up that EPP routes to rank 1.
	first := s.newKVReuse(requestWithSession(eppSessionToken(testPodNamespace, testPodName, 0)), chatBody(chatMessage("user", "hello")), reqcommon.APITypeChatCompletions)
	require.True(t, first.store(captureJSON(t, nixlDecodeResponse(t, chatMessage("assistant", "hi there")))))

	followUp := chatBody(chatMessage("user", "hello"), chatMessage("assistant", "hi there"), chatMessage("user", "more"))
	second := rankOne.newKVReuse(requestWithSession(eppSessionToken(testPodNamespace, testPodName, 1)), followUp, reqcommon.APITypeChatCompletions)
	require.NotNil(t, second, "the rank clone keeps the feature enabled")
	require.NotNil(t, second.take())
}

func TestKVReuseReplaysOnlyToARoutedFollowUp(t *testing.T) {
	s := newBidirectionalTestServer(t, 1)
	first := s.newKVReuse(requestWithSession(""), chatBody(chatMessage("user", "hello")), reqcommon.APITypeChatCompletions)
	require.True(t, first.store(captureJSON(t, nixlDecodeResponse(t, chatMessage("assistant", "hi there")))),
		"the first turn is cached although it carries no token")

	followUp := func() map[string]any {
		return chatBody(chatMessage("user", "hello"), chatMessage("assistant", "hi there"), chatMessage("user", "more"))
	}
	for name, token := range map[string]string{
		"no token":                    "",
		"a token for another pod":     eppSessionToken(testPodNamespace, "decode-1", 0),
		"a token for another cluster": eppSessionToken("other", testPodName, 0),
	} {
		require.Nil(t, s.newKVReuse(requestWithSession(token), followUp(), reqcommon.APITypeChatCompletions).take(), name)
	}
	require.NotNil(t, s.newKVReuse(requestWithSession(eppSessionToken(testPodNamespace, testPodName, 0)), followUp(), reqcommon.APITypeChatCompletions).take(),
		"the refused lookups left the entry in place")
}

func TestDropBidirectionalKVParamsRestoresThePlainPrefillRequest(t *testing.T) {
	kv := map[string]any{
		"do_remote_decode":  true,
		"do_remote_prefill": false,
		"remote_engine_id":  nil,
		"remote_block_ids":  nil,
		"remote_host":       nil,
		"remote_port":       nil,
	}
	cached, ok := completeKVParams(map[string]any{
		"do_remote_decode": true, "remote_block_ids": []any{[]any{1}}, "remote_engine_id": "e",
		"remote_request_id": "r", "remote_host": "h", "remote_port": json.Number("1"), "remote_num_tokens": json.Number("64"),
	})
	require.True(t, ok)
	injectBidirectionalKVParams(kv, cached)
	dropBidirectionalKVParams(kv)

	require.Equal(t, map[string]any{
		"do_remote_decode":  true,
		"do_remote_prefill": false,
		"remote_engine_id":  nil,
		"remote_block_ids":  nil,
		"remote_host":       nil,
		"remote_port":       nil,
	}, kv)
}
