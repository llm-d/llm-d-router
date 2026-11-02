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

// Package grpctest holds test scaffolding for components that speak gRPC
// over HTTP/2: h2c servers and clients, frame helpers, trailer writers, and
// a scripted fake vllm.Inference upstream. Only _test files import it.
package grpctest

import (
	"errors"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/protobuf/proto"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

// NewH2CServer starts an httptest server that accepts HTTP/1 and cleartext
// HTTP/2 (prior knowledge) and registers Close with t.
func NewH2CServer(t testing.TB, h http.Handler) *httptest.Server {
	t.Helper()
	srv := httptest.NewUnstartedServer(h)
	protocols := new(http.Protocols)
	protocols.SetHTTP1(true)
	protocols.SetUnencryptedHTTP2(true)
	srv.Config.Protocols = protocols
	srv.Start()
	t.Cleanup(srv.Close)
	return srv
}

// NewH2CClient returns an http.Client that speaks cleartext HTTP/2 only.
func NewH2CClient() *http.Client {
	protocols := new(http.Protocols)
	protocols.SetUnencryptedHTTP2(true)
	return &http.Client{Transport: &http.Transport{Protocols: protocols}}
}

// StartGRPCServer serves a grpc.Server on a loopback listener and returns
// its address. register adds the services. Stop runs at cleanup.
func StartGRPCServer(t testing.TB, register func(*grpc.Server)) string {
	t.Helper()
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	srv := grpc.NewServer()
	register(srv)
	go func() { _ = srv.Serve(ln) }()
	t.Cleanup(srv.Stop)
	return ln.Addr().String()
}

// Dial returns an insecure grpc.ClientConn to addr and closes it at cleanup.
func Dial(t testing.TB, addr string) *grpc.ClientConn {
	t.Helper()
	conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
	require.NoError(t, err)
	t.Cleanup(func() { _ = conn.Close() })
	return conn
}

// Frame marshals m into a 5-byte-prefixed gRPC frame.
func Frame(t testing.TB, m proto.Message) []byte {
	t.Helper()
	framed, err := grpcframe.MarshalProto(m)
	require.NoError(t, err)
	return framed
}

// FramePayload wraps a raw payload into a 5-byte-prefixed gRPC frame.
func FramePayload(t testing.TB, payload []byte) []byte {
	t.Helper()
	framed, err := grpcframe.FramePayload(payload)
	require.NoError(t, err)
	return framed
}

// Unmarshal parses exactly one frame into dst.
func Unmarshal(t testing.TB, framed []byte, dst proto.Message) {
	t.Helper()
	payload, err := grpcframe.ParsePayload(framed)
	require.NoError(t, err)
	require.NoError(t, proto.Unmarshal(payload, dst))
}

// ReadFrames reads r to the end and splits it into frames, each with its
// 5-byte header. It fails the test on malformed framing.
func ReadFrames(t testing.TB, r io.Reader) [][]byte {
	t.Helper()
	fr := grpcframe.NewReader(r, grpcframe.MaxPayloadLen)
	defer fr.Release()
	var frames [][]byte
	for {
		frame, err := fr.Next()
		if errors.Is(err, io.EOF) {
			return frames
		}
		require.NoError(t, err)
		frames = append(frames, append([]byte(nil), frame...))
	}
}

// ReadRequest reads the single framed message in r.Body into dst.
func ReadRequest(t testing.TB, r *http.Request, dst proto.Message) {
	t.Helper()
	framed, err := grpcframe.ReadMessage(r.Body, grpcframe.MaxPayloadLen)
	require.NoError(t, err)
	Unmarshal(t, framed, dst)
}

// WriteFrames writes a successful gRPC response: the content type, status
// 200, one frame per message, and an OK status trailer.
func WriteFrames(t testing.TB, w http.ResponseWriter, msgs ...proto.Message) {
	t.Helper()
	w.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
	w.WriteHeader(http.StatusOK)
	for _, m := range msgs {
		_, err := w.Write(Frame(t, m))
		require.NoError(t, err)
	}
	WriteStatusTrailer(w, codes.OK, "")
}

// WriteStatusTrailer sets the gRPC status as HTTP trailers. Call it after
// the body is written and before the handler returns.
func WriteStatusTrailer(w http.ResponseWriter, code codes.Code, msg string) {
	w.Header().Set(http.TrailerPrefix+grpcframe.HeaderStatus, strconv.Itoa(int(code)))
	if msg != "" {
		w.Header().Set(http.TrailerPrefix+grpcframe.HeaderMessage, grpcframe.EncodeStatusMessage(msg))
	}
}
