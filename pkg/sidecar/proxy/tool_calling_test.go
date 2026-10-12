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
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"

	"github.com/llm-d/llm-d-router/pkg/common/observability/toolcalling"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
	"github.com/llm-d/llm-d-router/pkg/common/routing"
)

func TestToolCallingPassthroughBody(t *testing.T) {
	const body = `{"tools":[],"tool_choice":"auto"}`
	for _, tt := range []struct {
		name    string
		length  int64
		read    int
		readErr error
		want    bool
	}{
		{name: "known length without EOF", length: int64(len(body)), read: len(body), want: true},
		{name: "unknown length with EOF", length: -1, read: len(body), readErr: io.EOF, want: true},
		{name: "zero length treated as unknown", read: len(body), readErr: io.EOF, want: true},
		{name: "unknown length without EOF", length: -1, read: len(body)},
		{name: "partial known length", length: int64(len(body)), read: 4},
		{name: "short body EOF", length: int64(len(body)) + 1, read: len(body), readErr: io.EOF},
		{name: "read failure with bytes", length: int64(len(body)), read: len(body), readErr: io.ErrUnexpectedEOF},
		{name: "canceled read with bytes", length: int64(len(body)), read: len(body), readErr: context.Canceled},
		{name: "unread body", length: int64(len(body))},
	} {
		t.Run(tt.name, func(t *testing.T) {
			closeErr := errors.New("close failure")
			recorder := tracetest.NewSpanRecorder()
			provider := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
			t.Cleanup(func() { require.NoError(t, provider.Shutdown(context.Background())) })
			ctx, span := provider.Tracer(tracerScope).Start(context.Background(), "forward_request")
			ctx = context.WithValue(ctx, toolCallingRequestSpanKey, span)
			source := &toolCallingTestBody{body: body[:tt.read], err: tt.readErr, closeErr: closeErr}
			r := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, nil).WithContext(ctx)
			r.Body, r.ContentLength = source, tt.length
			before := toolFieldCounters(t)
			observeToolCallingPassthrough(r, reqcommon.APITypeChatCompletions)
			require.Zero(t, source.reads, "observation must not pre-read")
			require.Equal(t, tt.length, r.ContentLength)
			if tt.read > 0 {
				buf := make([]byte, len(body))
				n, err := r.Body.Read(buf)
				require.Equal(t, body[:tt.read], string(buf[:n]))
				require.Equal(t, tt.readErr, err)
				_, _ = r.Body.Read(buf) // A second EOF must not duplicate metrics.
			}
			require.ErrorIs(t, r.Body.Close(), closeErr)
			require.Equal(t, 1, source.closes)
			want := map[[2]string]float64{}
			if tt.want {
				want[[2]string{string(toolcalling.FieldTools), string(toolcalling.FieldStatusPreserved)}] = 1
				want[[2]string{string(toolcalling.FieldToolChoice), string(toolcalling.FieldStatusPreserved)}] = 1
			}
			delta := map[[2]string]float64{}
			for key, value := range toolFieldCounters(t) {
				if change := value - before[key]; change != 0 {
					delta[key] = change
				}
			}
			require.Equal(t, want, delta)
			span.End()
			if tt.want {
				snapshot, err := toolcalling.CaptureRequestJSON(reqcommon.APITypeChatCompletions, []byte(body))
				require.NoError(t, err)
				statuses, err := toolcalling.CompareRequests(snapshot, snapshot)
				require.NoError(t, err)
				require.ElementsMatch(t, snapshot.SpanAttributes(statuses), recorder.Ended()[0].Attributes())
			} else {
				require.Empty(t, recorder.Ended()[0].Attributes())
			}
		})
	}
}

type toolCallingTestBody struct {
	body     string
	err      error
	closeErr error
	reads    int
	closes   int
}

func (b *toolCallingTestBody) Read(p []byte) (int, error) {
	b.reads++
	n := copy(p, b.body)
	b.body = b.body[n:]
	return n, b.err
}

func (b *toolCallingTestBody) Close() error {
	b.closes++
	return b.closeErr
}

func TestToolCallingPassthroughConcurrentClose(t *testing.T) {
	reader, writer := io.Pipe()
	r := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, nil)
	r.Body, r.ContentLength = reader, -1
	observeToolCallingPassthrough(r, reqcommon.APITypeChatCompletions)
	done := make(chan error, 1)
	go func() {
		_, err := io.Copy(io.Discard, r.Body)
		done <- err
	}()
	_, err := writer.Write([]byte(`{"tools":`))
	require.NoError(t, err)
	require.NoError(t, r.Body.Close())
	require.ErrorIs(t, <-done, io.ErrClosedPipe)
	require.NoError(t, writer.Close())
}

func TestToolCallingPassthroughSkipsUnsupportedOrEmptyBody(t *testing.T) {
	for _, api := range []reqcommon.APIType{reqcommon.APITypeChatCompletions, reqcommon.APITypeResponses, reqcommon.APITypeCompletions} {
		for _, body := range []io.ReadCloser{nil, http.NoBody, io.NopCloser(strings.NewReader(`{"tools":[]}`))} {
			if api == reqcommon.APITypeChatCompletions && body != nil && body != http.NoBody {
				continue
			}
			r := &http.Request{Body: body}
			observeToolCallingPassthrough(r, api)
			require.True(t, body == r.Body)
		}
	}
}

func TestToolCallingPassthroughMissingReplica(t *testing.T) {
	s := NewProxy(Config{})
	s.allowlistValidator = &AllowlistValidator{}
	s.decoderProxy = http.HandlerFunc(func(http.ResponseWriter, *http.Request) {
		t.Error("local decoder must not run for a missing replica")
	})
	source := &toolCallingTestBody{body: `{"tools":[]}`, err: io.EOF}
	r := httptest.NewRequest(http.MethodPost, reqcommon.PathChatCompletions, nil)
	r.Body = source
	r.Header.Set(routing.DataParallelEndpointHeader, "missing:8000")
	before := toolFieldCounters(t)
	rw := httptest.NewRecorder()
	s.disaggregatedPrefillHandler(reqcommon.APITypeChatCompletions)(rw, r)
	require.Equal(t, http.StatusBadRequest, rw.Code)
	require.Zero(t, source.reads)
	require.Same(t, source, r.Body)
	require.Equal(t, before, toolFieldCounters(t))
}

func BenchmarkToolCallingPassthrough(b *testing.B) {
	buf := make([]byte, 32768)
	for _, api := range []reqcommon.APIType{reqcommon.APITypeChatCompletions, reqcommon.APITypeMessages} {
		for _, size := range []int{1024, 1 << 20} {
			for _, toolRequest := range []bool{false, true} {
				body := toolCallingBenchmarkBody(b, api, size, toolRequest)
				for _, observe := range []bool{false, true} {
					b.Run(fmt.Sprintf("%s/%d/tools=%t/observe=%t", api, size, toolRequest, observe), func(b *testing.B) {
						decoder := http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
							read := 0
							for {
								n, err := r.Body.Read(buf)
								read += n
								if err == io.EOF {
									break
								}
								if err != nil {
									b.Fatal(err)
								}
							}
							if read != len(body) {
								b.Fatalf("read %d bytes, want %d", read, len(body))
							}
							if err := r.Body.Close(); err != nil {
								b.Fatal(err)
							}
						})
						b.ReportAllocs()
						b.SetBytes(int64(len(body)))
						for b.Loop() {
							r := &http.Request{Body: io.NopCloser(strings.NewReader(body)), ContentLength: int64(len(body))}
							if observe {
								observeToolCallingPassthrough(r, api)
							}
							decoder.ServeHTTP(nil, r)
						}
					})
				}
			}
		}
	}
}

// The body is already available to routing; only telemetry work is timed.
func BenchmarkToolCallingRequestDispatch(b *testing.B) {
	for _, api := range []reqcommon.APIType{reqcommon.APITypeChatCompletions, reqcommon.APITypeMessages} {
		for _, size := range []int{1024, 1 << 20} {
			for _, toolRequest := range []bool{false, true} {
				body := []byte(toolCallingBenchmarkBody(b, api, size, toolRequest))
				for _, dispatches := range []int{1, 2} {
					b.Run(fmt.Sprintf("%s/%d/tools=%t/dispatches=%d", api, size, toolRequest, dispatches), func(b *testing.B) {
						r := &http.Request{}
						b.ReportAllocs()
						b.SetBytes(int64(len(body)))
						for b.Loop() {
							fields, err := captureInferenceToolCallingRequest(r, api, body)
							if err != nil {
								b.Fatal(err)
							}
							for range dispatches {
								if _, err := fields.compareDispatched(body); err != nil {
									b.Fatal(err)
								}
							}
							fields.report(r.Context())
						}
					})
				}
			}
		}
	}
}

func toolCallingBenchmarkBody(b *testing.B, api reqcommon.APIType, size int, toolRequest bool) string {
	b.Helper()
	body := map[string]any{reqcommon.FieldMessages: []any{map[string]any{"role": "user", "content": strings.Repeat("x", size)}}}
	if toolRequest {
		schema := map[string]any{"type": "object", "properties": map[string]any{"city": map[string]any{"type": "string"}}}
		if api == reqcommon.APITypeChatCompletions {
			body[string(toolcalling.FieldTools)] = []any{map[string]any{"type": "function", "function": map[string]any{"name": "weather", "parameters": schema}}}
			body[string(toolcalling.FieldToolChoice)] = "auto"
		} else {
			body[string(toolcalling.FieldTools)] = []any{map[string]any{"name": "weather", "input_schema": schema}}
			body[string(toolcalling.FieldToolChoice)] = map[string]any{"type": "auto"}
		}
	}
	encoded, err := json.Marshal(body)
	require.NoError(b, err)
	return string(encoded)
}

func TestToolCallingJSONStream(t *testing.T) {
	for _, body := range []string{
		`{}`, ` { "tools" : null } `,
		`{"tools":[{"a":9007199254740993,"b":-1.20e+3}],"tool_choice":"auto","parallel_tool_calls":false,"response_format":{}}`,
		`{"messages":[{"tools":[],"content":"tools \\\" \\u1234"}],"to\u006fls":[]}`,
		`{"tools":[1],"tools":null,"tool_choice":true}`,
		`{"messages":["` + strings.Repeat("x", 1<<20) + `"],"tools":[]}`,
		`{"` + strings.Repeat("x", 1<<20) + `":1,"tool_choice":"none"}`,
		`{"unused":[{},[],true,false,null,-0,1.5e-2],"tools":[]}`,
		`{"\u0070\u0061\u0072\u0061\u006c\u006c\u0065\u006c\u005f\u0074\u006f\u006f\u006c\u005f\u0063\u0061\u006c\u006c\u0073":true}`,
		`{"tools":["` + strings.Repeat("x", 128<<10) + `"]}`,
		`{"unused":` + strings.Repeat("9", 128<<10) + `,"tools":[]}`,
		`{"unused":` + strings.Repeat("[", 9999) + `0` + strings.Repeat("]", 9999) + `,"tools":[]}`,
		`{"unused":` + strings.Repeat("[", 10000) + `0` + strings.Repeat("]", 10000) + `,"tools":[]}`,
		`{"tools":[],"unused":1.0E+20}`, `{"unused":"\ud800","tools":[]}`,
		`{"unused":[],"tools":[],}`, `{"unused":[}`, `{"unused":{]}`,
		`{"unused":truefalse}`, `{"unused":false null}`, `{"unused":-}`,
		`{"unused":1.}`, `{"unused":1e+}`, `{"unused":+1}`, `{"unused":NaN}`,
		"{\"unused\":\"\x00\",\"tools\":[]}",
		`{"tools":`, `{"tools":[]} garbage`, `{"unused":"bad\q","tools":[]}`,
		`{"unused":"bad\u12xz","tools":[]}`, `{"unused":[1,],"tools":[]}`,
		`{"unused":{"a":1,},"tools":[]}`, `{"unused":01,"tools":[]}`,
		`{"unused":1e,"tools":[]}`, `{"unused":` + "true"[:3] + `,"tools":[]}`,
		`[]`, `null`, `{} {}`, `{"tools":[],"unfinished":"`, "",
	} {
		for _, api := range []reqcommon.APIType{reqcommon.APITypeChatCompletions, reqcommon.APITypeMessages} {
			before, wantErr := toolcalling.CaptureRequestJSON(api, []byte(body))
			for _, chunkSize := range []int{1, 7, 32768} {
				stream := newToolCallingJSONStream(api)
				for offset := 0; offset < len(body); offset += chunkSize {
					stream.feed([]byte(body[offset:min(offset+chunkSize, len(body))]))
				}
				raw, ok := stream.finish()
				require.Equal(t, wantErr == nil, ok, "body prefix %q, chunk %d", body[:min(60, len(body))], chunkSize)
				if !ok {
					continue
				}
				var fields map[string]json.RawMessage
				require.NoError(t, json.Unmarshal(raw, &fields))
				for field := range fields {
					require.Contains(t, []string{string(toolcalling.FieldTools), string(toolcalling.FieldToolChoice), string(toolcalling.FieldParallelToolCalls), string(toolcalling.FieldResponseFormat)}, field)
				}
				after, err := toolcalling.CaptureRequestJSON(api, raw)
				require.NoError(t, err)
				statuses, err := toolcalling.CompareRequests(before, after)
				require.NoError(t, err)
				for _, status := range statuses {
					require.Equal(t, toolcalling.FieldStatusPreserved, status.Status)
				}
				require.Equal(t, before.Summary(), after.Summary())
			}
		}
	}
}

func TestToolCallingJSONStreamSkipsUnsupportedAPI(t *testing.T) {
	stream := newToolCallingJSONStream(reqcommon.APITypeResponses)
	stream.feed([]byte(`{"tools":[]}`))
	_, ok := stream.finish()
	require.False(t, ok)
}

func FuzzToolCallingJSONStream(f *testing.F) {
	for _, body := range []string{
		`{}`, `{"tools":[],"tool_choice":"auto"}`, `{"text":"escaped\"","tools":null}`,
		`{"a":[1,true,null]}`, `{"tools":[]}x`, `{"to\u006fls":[{"x":-0.123E+4}]}`,
		`{"tools":[],"tools":null}`, `{"text":"\ud800\udc00","tool_choice":{"name":"f"}}`,
	} {
		f.Add(body, uint8(1))
	}
	f.Fuzz(func(t *testing.T, body string, split uint8) {
		stream := newToolCallingJSONStream(reqcommon.APITypeChatCompletions)
		chunkSize := int(split) + 1
		for offset := 0; offset < len(body); offset += chunkSize {
			stream.feed([]byte(body[offset:min(offset+chunkSize, len(body))]))
		}
		raw, ok := stream.finish()
		before, err := toolcalling.CaptureRequestJSON(reqcommon.APITypeChatCompletions, []byte(body))
		require.Equal(t, err == nil, ok)
		if ok {
			after, err := toolcalling.CaptureRequestJSON(reqcommon.APITypeChatCompletions, raw)
			require.NoError(t, err)
			statuses, err := toolcalling.CompareRequests(before, after)
			require.NoError(t, err)
			for _, status := range statuses {
				require.Equal(t, toolcalling.FieldStatusPreserved, status.Status)
			}
		}
	})
}

func BenchmarkToolCallingJSONStream(b *testing.B) {
	for _, size := range []int{1024, 1 << 20} {
		body, err := json.Marshal(map[string]string{"prompt": strings.Repeat("x", size)})
		require.NoError(b, err)
		b.Run(strconv.Itoa(size), func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(body)))
			for b.Loop() {
				stream := newToolCallingJSONStream(reqcommon.APITypeChatCompletions)
				for offset := 0; offset < len(body); offset += 32768 {
					stream.feed(body[offset:min(offset+32768, len(body))])
				}
				_, ok := stream.finish()
				if !ok {
					b.Fatal("valid body was not captured")
				}
			}
		})
	}
}

func TestCaptureInferenceToolCallingRequest(t *testing.T) {
	const original = `{"tools":[],"tool_choice":"auto"}`
	for _, api := range []reqcommon.APIType{reqcommon.APITypeChatCompletions, reqcommon.APITypeMessages} {
		t.Run(api.String(), func(t *testing.T) {
			baseline, err := captureToolCallingRequest(api, []byte(original))
			require.NoError(t, err)
			baseline.rejectFields(toolcalling.FieldTools)
			r := httptest.NewRequest(http.MethodPost, "/", nil)
			r = r.WithContext(context.WithValue(r.Context(), toolCallingBaselineKey, baseline))
			// A cloned handoff uses the snapshot, not the modified body or prior outcomes.
			first, err := captureInferenceToolCallingRequest(r.Clone(r.Context()), api, []byte(`{`))
			require.NoError(t, err)
			require.Empty(t, first.fieldStatuses())
			_, err = first.compareDispatched([]byte(`{"tools":[],"tool_choice":"none"}`))
			require.NoError(t, err)
			assertToolCallingStatuses(t, first.fieldStatuses(), map[toolcalling.Field]toolcalling.FieldStatusValue{
				toolcalling.FieldTools: toolcalling.FieldStatusPreserved, toolcalling.FieldToolChoice: toolcalling.FieldStatusChanged,
			})
			second, err := captureInferenceToolCallingRequest(r, api, []byte(`{}`))
			require.NoError(t, err)
			require.NotSame(t, first, second)
			require.Empty(t, second.fieldStatuses())
			_, err = second.compareDispatched([]byte(original))
			require.NoError(t, err)
			assertToolCallingStatuses(t, second.fieldStatuses(), map[toolcalling.Field]toolcalling.FieldStatusValue{
				toolcalling.FieldTools: toolcalling.FieldStatusPreserved, toolcalling.FieldToolChoice: toolcalling.FieldStatusPreserved,
			})
			assertToolCallingStatuses(t, baseline.fieldStatuses(), map[toolcalling.Field]toolcalling.FieldStatusValue{
				toolcalling.FieldTools: toolcalling.FieldStatusRejected,
			})
		})
	}
	t.Run("no handoff", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodPost, "/", nil)
		request, err := captureInferenceToolCallingRequest(r, reqcommon.APITypeChatCompletions, []byte(original))
		require.NoError(t, err)
		_, err = request.compareDispatched([]byte(`{}`))
		require.NoError(t, err)
		assertToolCallingStatuses(t, request.fieldStatuses(), map[toolcalling.Field]toolcalling.FieldStatusValue{
			toolcalling.FieldTools: toolcalling.FieldStatusDropped, toolcalling.FieldToolChoice: toolcalling.FieldStatusDropped,
		})
	})
	t.Run("failed capture stays disabled", func(t *testing.T) {
		r := httptest.NewRequest(http.MethodPost, "/", nil)
		r = r.WithContext(context.WithValue(r.Context(), toolCallingBaselineKey, (*toolCallingRequest)(nil)))
		request, err := captureInferenceToolCallingRequest(r, reqcommon.APITypeChatCompletions, []byte(original))
		require.Error(t, err)
		require.Nil(t, request)
	})
	t.Run("API mismatch", func(t *testing.T) {
		baseline, err := captureToolCallingRequest(reqcommon.APITypeChatCompletions, []byte(original))
		require.NoError(t, err)
		r := httptest.NewRequest(http.MethodPost, "/", nil)
		r = r.WithContext(context.WithValue(r.Context(), toolCallingBaselineKey, baseline))
		request, err := captureInferenceToolCallingRequest(r, reqcommon.APITypeMessages, []byte(original))
		require.Error(t, err)
		require.Nil(t, request)
	})
}

func TestToolCallingRequestComparisons(t *testing.T) {
	tests := []struct {
		name   string
		api    reqcommon.APIType
		before string
		after  string
		want   map[toolcalling.Field]toolcalling.FieldStatusValue
	}{
		{
			name:   "chat semantic equality",
			before: `{"tools":[{"b":1e2,"a":9007199254740993}],"tool_choice":"auto","parallel_tool_calls":false,"response_format":{"type":"json_object"}}`,
			after:  `{"response_format":{"type":"json_object"},"parallel_tool_calls":false,"tool_choice":"auto","tools":[{"a":9007199254740993,"b":100.0}]}`,
			want: map[toolcalling.Field]toolcalling.FieldStatusValue{
				toolcalling.FieldTools: toolcalling.FieldStatusPreserved, toolcalling.FieldToolChoice: toolcalling.FieldStatusPreserved,
				toolcalling.FieldParallelToolCalls: toolcalling.FieldStatusPreserved, toolcalling.FieldResponseFormat: toolcalling.FieldStatusPreserved,
			},
		},
		{
			name: "changed and dropped", before: `{"tools":[1,2],"tool_choice":"auto"}`, after: `{"tools":[2,1]}`,
			want: map[toolcalling.Field]toolcalling.FieldStatusValue{toolcalling.FieldTools: toolcalling.FieldStatusChanged, toolcalling.FieldToolChoice: toolcalling.FieldStatusDropped},
		},
		{
			name: "absent to null", before: `{}`, after: `{"tools":null}`,
			want: map[toolcalling.Field]toolcalling.FieldStatusValue{toolcalling.FieldTools: toolcalling.FieldStatusChanged},
		},
		{
			name: "null to absent", before: `{"tools":null}`, after: `{}`,
			want: map[toolcalling.Field]toolcalling.FieldStatusValue{toolcalling.FieldTools: toolcalling.FieldStatusDropped},
		},
		{
			name: "explicit null preserved", before: `{"tools":null}`, after: `{"tools":null}`,
			want: map[toolcalling.Field]toolcalling.FieldStatusValue{toolcalling.FieldTools: toolcalling.FieldStatusPreserved},
		},
		{
			name: "messages supports only tools and choice", api: reqcommon.APITypeMessages,
			before: `{"tools":[],"tool_choice":{"type":"any"},"parallel_tool_calls":false,"response_format":null}`,
			after:  `{"tools":[],"tool_choice":{"type":"auto"}}`,
			want:   map[toolcalling.Field]toolcalling.FieldStatusValue{toolcalling.FieldTools: toolcalling.FieldStatusPreserved, toolcalling.FieldToolChoice: toolcalling.FieldStatusChanged},
		},
		{name: "non-tool", before: `{"messages":[]}`, after: `{"messages":[],"max_tokens":1}`},
		{name: "nested fields", before: `{"messages":[{"tools":[]}]}`, after: `{"messages":[]}`},
		{name: "messages unsupported fields", api: reqcommon.APITypeMessages, before: `{"parallel_tool_calls":true}`, after: `{"parallel_tool_calls":false}`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			request, err := captureToolCallingRequest(tt.api, []byte(tt.before))
			require.NoError(t, err)
			_, err = request.compareDispatched([]byte(tt.after))
			require.NoError(t, err)
			assertToolCallingStatuses(t, request.fieldStatuses(), tt.want)
		})
	}
}

func TestToolCallingRequestAggregatesDispatches(t *testing.T) {
	request, err := captureToolCallingRequest(reqcommon.APITypeChatCompletions, []byte(`{"tools":[1],"tool_choice":"auto"}`))
	require.NoError(t, err)
	// Preparing a request without dispatching it supplies no preservation evidence.
	require.Empty(t, request.fieldStatuses())
	for _, body := range []string{
		`{"tools":[1],"tool_choice":"auto"}`,
		`{"tools":[2],"tool_choice":"auto"}`,
		`{"tool_choice":"auto"}`,
		`{"tools":[1],"tool_choice":"auto"}`,
		`{"tools":[1],"tool_choice":"auto"}`,
	} {
		_, err := request.compareDispatched([]byte(body))
		require.NoError(t, err)
	}
	want := map[toolcalling.Field]toolcalling.FieldStatusValue{
		toolcalling.FieldTools: toolcalling.FieldStatusDropped, toolcalling.FieldToolChoice: toolcalling.FieldStatusPreserved,
	}
	assertToolCallingStatuses(t, request.fieldStatuses(), want)
	// Callers cannot change the aggregate through a returned slice.
	statuses := request.fieldStatuses()
	statuses[0].Status = toolcalling.FieldStatusChanged
	assertToolCallingStatuses(t, request.fieldStatuses(), want)
	stage, err := request.compareDispatched([]byte(`{"tools":[1],"tool_choice":"auto"}`))
	require.NoError(t, err)
	require.Equal(t, toolcalling.FieldStatusPreserved, stage[0].Status)
	assertToolCallingStatuses(t, request.fieldStatuses(), want)
}

func TestToolCallingRequestExplicitRejection(t *testing.T) {
	request, err := captureToolCallingRequest(reqcommon.APITypeChatCompletions, []byte(`{"tools":[],"tool_choice":null}`))
	require.NoError(t, err)
	_, err = request.compareDispatched([]byte(`{}`))
	require.NoError(t, err)
	request.rejectFields(toolcalling.FieldTools, toolcalling.FieldToolChoice)
	_, err = request.compareDispatched([]byte(`{"tools":[],"tool_choice":null}`))
	require.NoError(t, err)
	assertToolCallingStatuses(t, request.fieldStatuses(), map[toolcalling.Field]toolcalling.FieldStatusValue{
		toolcalling.FieldTools: toolcalling.FieldStatusRejected, toolcalling.FieldToolChoice: toolcalling.FieldStatusRejected,
	})
	plain, err := captureToolCallingRequest(reqcommon.APITypeMessages, []byte(`{"messages":[]}`))
	require.NoError(t, err)
	plain.rejectFields(toolcalling.FieldTools)
	require.Empty(t, plain.fieldStatuses())
}

func TestToolCallingRequestCaptureFailures(t *testing.T) {
	for _, body := range []string{`{"tools":`, `null`, `[]`, ``} {
		request, err := captureToolCallingRequest(reqcommon.APITypeChatCompletions, []byte(body))
		require.Error(t, err)
		require.Nil(t, request)
		_, err = request.compareDispatched([]byte(`{"tools":[]}`))
		require.NoError(t, err)
		request.rejectFields(toolcalling.FieldTools)
		require.Empty(t, request.fieldStatuses())
	}
	request, err := captureToolCallingRequest(reqcommon.APITypeChatCompletions, []byte(`{"tools":[]}`))
	require.NoError(t, err)
	_, err = request.compareDispatched([]byte(`{"tools":`))
	require.Error(t, err)
	require.Empty(t, request.fieldStatuses())
	_, err = request.compareDispatched([]byte(`{"tools":[]}`))
	require.NoError(t, err)
	_, err = request.compareDispatched([]byte(`null`))
	require.Error(t, err)
	assertToolCallingStatuses(t, request.fieldStatuses(), map[toolcalling.Field]toolcalling.FieldStatusValue{toolcalling.FieldTools: toolcalling.FieldStatusPreserved})
}

func TestToolCallingRequestUnsupportedAPI(t *testing.T) {
	request, err := captureToolCallingRequest(reqcommon.APITypeResponses, []byte(`{"tools":[]}`))
	require.Error(t, err)
	require.Nil(t, request)
}

func TestToolCallingRequestRejectsOnlyIdentifiedFields(t *testing.T) {
	for _, tt := range []struct {
		name   string
		body   string
		fields []toolcalling.Field
		want   map[toolcalling.Field]toolcalling.FieldStatusValue
	}{
		{name: "unspecified"},
		{name: "absent field", body: `{"tool_choice":null}`, fields: []toolcalling.Field{toolcalling.FieldTools}},
		{name: "unsupported field", fields: []toolcalling.Field{toolcalling.FieldResponseFormat}},
		{name: "one identified field", fields: []toolcalling.Field{toolcalling.FieldTools}, want: map[toolcalling.Field]toolcalling.FieldStatusValue{toolcalling.FieldTools: toolcalling.FieldStatusRejected}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			body := tt.body
			if body == "" {
				body = `{"tools":[],"tool_choice":null,"response_format":null}`
			}
			request, err := captureToolCallingRequest(reqcommon.APITypeMessages, []byte(body))
			require.NoError(t, err)
			request.rejectFields(tt.fields...)
			assertToolCallingStatuses(t, request.fieldStatuses(), tt.want)
		})
	}
}

func TestToolCallingRequestStatusPrecedence(t *testing.T) {
	statuses := []toolcalling.FieldStatusValue{toolcalling.FieldStatusPreserved, toolcalling.FieldStatusChanged, toolcalling.FieldStatusDropped, toolcalling.FieldStatusRejected}
	for i, first := range statuses {
		for j, second := range statuses {
			t.Run(string(first)+"/"+string(second), func(t *testing.T) {
				request := &toolCallingRequest{}
				for _, status := range []toolcalling.FieldStatusValue{first, second} {
					request.mergeStatuses([]toolcalling.FieldStatus{{Field: toolcalling.FieldTools, Status: status, Observed: true}})
				}
				assertToolCallingStatuses(t, request.fieldStatuses(), map[toolcalling.Field]toolcalling.FieldStatusValue{toolcalling.FieldTools: statuses[max(i, j)]})
			})
		}
	}
}

func TestToolCallingRequestOwnsSnapshot(t *testing.T) {
	body := []byte(`{"tools":[1]}`)
	request, err := captureToolCallingRequest(reqcommon.APITypeChatCompletions, body)
	require.NoError(t, err)
	copy(body, `{"tools":[2]}`)
	_, err = request.compareDispatched(body)
	require.NoError(t, err)
	assertToolCallingStatuses(t, request.fieldStatuses(), map[toolcalling.Field]toolcalling.FieldStatusValue{toolcalling.FieldTools: toolcalling.FieldStatusChanged})
}

func assertToolCallingStatuses(t *testing.T, statuses []toolcalling.FieldStatus, want map[toolcalling.Field]toolcalling.FieldStatusValue) {
	t.Helper()
	require.Len(t, statuses, len(want))
	seen := make(map[toolcalling.Field]bool, len(statuses))
	for _, result := range statuses {
		require.True(t, result.Observed)
		require.False(t, seen[result.Field], "duplicate outcome for %s", result.Field)
		seen[result.Field] = true
		require.Contains(t, want, result.Field)
		require.Equal(t, want[result.Field], result.Status)
	}
}
