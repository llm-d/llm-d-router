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

package sessionid_test

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	fwkplugin "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/plugin"
	fwksched "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/scheduling"
	attrsession "github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/datalayer/attribute/session"
	"github.com/llm-d/llm-d-router/pkg/epp/framework/plugins/requestcontrol/dataproducer/sessionid"
)

func mustFactory(t *testing.T, params string) *sessionid.Producer {
	t.Helper()
	plg, err := sessionid.Factory("session-id-producer", fwkplugin.StrictDecoder(json.RawMessage(params)), nil)
	require.NoError(t, err)
	p, ok := plg.(*sessionid.Producer)
	require.True(t, ok, "factory must return *Producer")
	return p
}

func TestFactory_Validation(t *testing.T) {
	t.Parallel()

	const validationErr = "requires exactly one of headerName or cookieName"

	tests := []struct {
		name      string
		params    json.RawMessage
		wantErr   bool
		errSubstr string
	}{
		{name: "header only", params: json.RawMessage(`{"headerName":"x-session-id"}`)},
		{name: "cookie only", params: json.RawMessage(`{"cookieName":"llm-d-session"}`)},
		{name: "header normalized", params: json.RawMessage(`{"headerName":" X-Session-ID "}`)},
		{name: "empty object", params: json.RawMessage(`{}`), wantErr: true, errSubstr: validationErr},
		{name: "both set", params: json.RawMessage(`{"headerName":"x","cookieName":"y"}`), wantErr: true, errSubstr: validationErr},
		{name: "empty strings", params: json.RawMessage(`{"headerName":"","cookieName":""}`), wantErr: true, errSubstr: validationErr},
		{name: "invalid json", params: json.RawMessage(`not-json`), wantErr: true, errSubstr: "failed to parse"},
		{name: "unknown field", params: json.RawMessage(`{"headerName":"x","other":"y"}`), wantErr: true, errSubstr: "failed to parse"},
		{name: "nil raw message", params: nil, wantErr: true, errSubstr: validationErr},
		{name: "zero-length raw message", params: json.RawMessage{}, wantErr: true, errSubstr: validationErr},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			_, err := sessionid.Factory("p", fwkplugin.StrictDecoder(tc.params), nil)
			if tc.wantErr {
				require.Error(t, err)
				assert.Contains(t, err.Error(), tc.errSubstr)
				return
			}
			assert.NoError(t, err)
		})
	}
}

func TestProduce_HeaderMode(t *testing.T) {
	t.Parallel()

	producer := mustFactory(t, `{"headerName":"x-session-id"}`)

	tests := []struct {
		name    string
		headers map[string]string
		want    string
	}{
		{
			name:    "value present",
			headers: map[string]string{"x-session-id": "user-42"},
			want:    "user-42",
		},
		{
			name:    "value trimmed",
			headers: map[string]string{"x-session-id": "  user-42  "},
			want:    "user-42",
		},
		{
			name:    "header absent",
			headers: map[string]string{"other": "irrelevant"},
		},
		{
			name:    "value empty",
			headers: map[string]string{"x-session-id": ""},
		},
		{
			name: "headers nil",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			req := &fwksched.InferenceRequest{Headers: tc.headers}

			err := producer.Produce(context.Background(), req, nil)
			require.NoError(t, err)

			got, ok := attrsession.ReadSessionID(req)
			if tc.want == "" {
				assert.False(t, ok, "no session id should be published")
				return
			}
			assert.True(t, ok)
			assert.Equal(t, tc.want, string(got))
		})
	}
}

func TestProduce_CookieMode(t *testing.T) {
	t.Parallel()

	producer := mustFactory(t, `{"cookieName":"llm-d-session"}`)

	tests := []struct {
		name   string
		cookie string
		want   string
	}{
		{
			name:   "single cookie",
			cookie: "llm-d-session=abc123",
			want:   "abc123",
		},
		{
			name:   "named cookie among others",
			cookie: "csrf=xyz; llm-d-session=abc123; theme=dark",
			want:   "abc123",
		},
		{
			name:   "name not present",
			cookie: "csrf=xyz; theme=dark",
		},
		{
			name:   "header empty",
			cookie: "",
		},
		{
			name:   "malformed pair",
			cookie: "no-equals; llm-d-session=abc",
			want:   "abc",
		},
		{
			name:   "value empty",
			cookie: "llm-d-session=",
		},
		{
			name:   "value trimmed",
			cookie: "llm-d-session= abc123 ; other=v",
			want:   "abc123",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			req := &fwksched.InferenceRequest{
				Headers: map[string]string{"cookie": tc.cookie},
			}

			err := producer.Produce(context.Background(), req, nil)
			require.NoError(t, err)

			got, ok := attrsession.ReadSessionID(req)
			if tc.want == "" {
				assert.False(t, ok)
				return
			}
			assert.True(t, ok)
			assert.Equal(t, tc.want, string(got))
		})
	}
}

func TestProduce_NilRequest(t *testing.T) {
	t.Parallel()

	producer := mustFactory(t, `{"headerName":"x-session-id"}`)
	err := producer.Produce(context.Background(), nil, nil)
	require.NoError(t, err)
}

func TestProduce_NoSessionDoesNotAllocateStore(t *testing.T) {
	t.Parallel()

	producer := mustFactory(t, `{"headerName":"x-session-id"}`)
	req := &fwksched.InferenceRequest{}

	require.NoError(t, producer.Produce(context.Background(), req, nil))
	assert.Empty(t, req.AttributeKeys(), "no extraction should leave the store unallocated")
}

func TestProduces_DeclaresKey(t *testing.T) {
	t.Parallel()

	producer := mustFactory(t, `{"headerName":"x"}`)
	produces := producer.Produces()
	expectedKey := attrsession.SessionIDDataKey.WithNonEmptyProducerName("session-id-producer")
	_, ok := produces[expectedKey]
	assert.True(t, ok, "Produces() must declare %v", expectedKey)
}
