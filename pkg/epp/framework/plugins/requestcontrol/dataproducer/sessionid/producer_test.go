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
	"net"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	k8stypes "k8s.io/apimachinery/pkg/types"

	fwkdl "github.com/llm-d/llm-d-router/pkg/epp/framework/interface/datalayer"
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
		{name: "with binding tuning", params: json.RawMessage(`{"headerName":"x","lruSize":10,"ttl":"5m"}`)},
		{name: "empty object", params: json.RawMessage(`{}`), wantErr: true, errSubstr: validationErr},
		{name: "both set", params: json.RawMessage(`{"headerName":"x","cookieName":"y"}`), wantErr: true, errSubstr: validationErr},
		{name: "empty strings", params: json.RawMessage(`{"headerName":"","cookieName":""}`), wantErr: true, errSubstr: validationErr},
		{name: "negative lru size", params: json.RawMessage(`{"headerName":"x","lruSize":-1}`), wantErr: true, errSubstr: "lruSize"},
		{name: "zero ttl", params: json.RawMessage(`{"headerName":"x","ttl":"0s"}`), wantErr: true, errSubstr: "ttl"},
		{name: "negative ttl", params: json.RawMessage(`{"headerName":"x","ttl":"-1m"}`), wantErr: true, errSubstr: "ttl"},
		{name: "unparsable ttl", params: json.RawMessage(`{"headerName":"x","ttl":"not-a-duration"}`), wantErr: true, errSubstr: "invalid ttl"},
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
	for _, key := range []fwkplugin.DataKey{
		attrsession.SessionIDDataKey.WithNonEmptyProducerName("session-id-producer"),
		attrsession.BoundEndpointDataKey.WithNonEmptyProducerName("session-id-producer"),
	} {
		_, ok := produces[key]
		assert.True(t, ok, "Produces() must declare %v", key)
	}
}

// requestWithSession is a small helper for binding tests; the producer reads
// session IDs from request headers and never panics on a nil Headers map.
func requestWithSession(id string) *fwksched.InferenceRequest {
	return &fwksched.InferenceRequest{Headers: map[string]string{"x-session-id": id}}
}

func endpointFor(name, address, port string) fwksched.Endpoint {
	return fwksched.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: k8stypes.NamespacedName{Namespace: "default", Name: name},
			Address:        address,
			Port:           port,
		},
		&fwkdl.Metrics{},
		nil,
	)
}

func boundTo(address, port string) attrsession.BoundEndpoint {
	return attrsession.BoundEndpoint(net.JoinHostPort(address, port))
}

func schedulingResultFor(profile string, endpoint fwksched.Endpoint) *fwksched.SchedulingResult {
	return &fwksched.SchedulingResult{
		PrimaryProfileName: profile,
		ProfileResults: map[string]*fwksched.ProfileRunResult{
			profile: {TargetEndpoints: []fwksched.Endpoint{endpoint}},
		},
	}
}

func TestPreRequestThenProducePublishesBinding(t *testing.T) {
	t.Parallel()

	producer := mustFactory(t, `{"headerName":"x-session-id"}`)
	bindReq := requestWithSession("session-A")
	producer.PreRequest(context.Background(), bindReq, schedulingResultFor("default", endpointFor("pod-1", "10.0.0.1", "8080")))

	lookup := requestWithSession("session-A")
	require.NoError(t, producer.Produce(context.Background(), lookup, nil))

	got, ok := fwksched.ReadRequestAttribute[attrsession.BoundEndpoint](
		lookup,
		attrsession.BoundEndpointDataKey.WithNonEmptyProducerName("session-id-producer").String(),
	)
	require.True(t, ok)
	assert.Equal(t, boundTo("10.0.0.1", "8080"), got)
}

func TestPreRequestIgnoresMissingSession(t *testing.T) {
	t.Parallel()

	producer := mustFactory(t, `{"headerName":"x-session-id"}`)
	producer.PreRequest(
		context.Background(),
		&fwksched.InferenceRequest{}, // no session header
		schedulingResultFor("default", endpointFor("pod-1", "10.0.0.1", "8080")),
	)

	// A subsequent Produce on a different session must not see the (non-existent) binding.
	lookup := requestWithSession("session-A")
	require.NoError(t, producer.Produce(context.Background(), lookup, nil))
	_, ok := fwksched.ReadRequestAttribute[attrsession.BoundEndpoint](
		lookup,
		attrsession.BoundEndpointDataKey.WithNonEmptyProducerName("session-id-producer").String(),
	)
	assert.False(t, ok)
}

func TestPreRequestIgnoresEmptyResult(t *testing.T) {
	t.Parallel()

	producer := mustFactory(t, `{"headerName":"x-session-id"}`)
	bindReq := requestWithSession("session-A")
	producer.PreRequest(context.Background(), bindReq, &fwksched.SchedulingResult{
		PrimaryProfileName: "default",
		ProfileResults:     map[string]*fwksched.ProfileRunResult{},
	})

	lookup := requestWithSession("session-A")
	require.NoError(t, producer.Produce(context.Background(), lookup, nil))
	_, ok := fwksched.ReadRequestAttribute[attrsession.BoundEndpoint](
		lookup,
		attrsession.BoundEndpointDataKey.WithNonEmptyProducerName("session-id-producer").String(),
	)
	assert.False(t, ok)
}

func TestBindingExpiresAfterTTL(t *testing.T) {
	t.Parallel()

	producer := mustFactory(t, `{"headerName":"x-session-id","ttl":"50ms"}`)
	bind := requestWithSession("session-A")
	producer.PreRequest(context.Background(), bind, schedulingResultFor("default", endpointFor("pod-1", "10.0.0.1", "8080")))

	time.Sleep(120 * time.Millisecond)

	lookup := requestWithSession("session-A")
	require.NoError(t, producer.Produce(context.Background(), lookup, nil))
	_, ok := fwksched.ReadRequestAttribute[attrsession.BoundEndpoint](
		lookup,
		attrsession.BoundEndpointDataKey.WithNonEmptyProducerName("session-id-producer").String(),
	)
	assert.False(t, ok, "binding should have expired")
}

func TestBindingsEvictedAtCapacity(t *testing.T) {
	t.Parallel()

	producer := mustFactory(t, `{"headerName":"x-session-id","lruSize":2}`)

	bind := func(session, endpoint, address string) {
		producer.PreRequest(
			context.Background(),
			requestWithSession(session),
			schedulingResultFor("default", endpointFor(endpoint, address, "8080")),
		)
	}
	bind("session-0", "pod-1", "10.0.0.1")
	bind("session-1", "pod-2", "10.0.0.2")
	bind("session-2", "pod-3", "10.0.0.3") // evicts session-0

	for i, session := range []string{"session-0", "session-1", "session-2"} {
		lookup := requestWithSession(session)
		require.NoError(t, producer.Produce(context.Background(), lookup, nil))
		_, ok := fwksched.ReadRequestAttribute[attrsession.BoundEndpoint](
			lookup,
			attrsession.BoundEndpointDataKey.WithNonEmptyProducerName("session-id-producer").String(),
		)
		if i == 0 {
			assert.False(t, ok, "expected %s to be evicted", session)
		} else {
			assert.True(t, ok, "expected %s to be present", session)
		}
	}
}
