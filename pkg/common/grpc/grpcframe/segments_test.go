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

package grpcframe

import (
	"bytes"
	"errors"
	"io"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestSegmentsReader(t *testing.T) {
	segs := [][]byte{[]byte("head"), nil, []byte("-payload-"), {}, []byte("tail")}
	want := bytes.Join(segs, nil)
	require.Equal(t, int64(len(want)), SegmentsLen(segs))

	for _, chunk := range []int{1, 3, 7, 64} {
		r := NewSegmentsReader(segs)
		var out bytes.Buffer
		buf := make([]byte, chunk)
		for {
			n, err := r.Read(buf)
			out.Write(buf[:n])
			if errors.Is(err, io.EOF) {
				break
			}
			require.NoError(t, err)
		}
		require.NoError(t, r.Close())
		require.Equal(t, want, out.Bytes(), "chunk %d", chunk)
	}

	n, err := NewSegmentsReader(nil).Read(make([]byte, 4))
	require.Zero(t, n)
	require.ErrorIs(t, err, io.EOF)
}

func TestSetSegmentedBody(t *testing.T) {
	head, payload := []byte{0, 0, 0, 0, 3}, []byte("abc")
	req, err := http.NewRequest(http.MethodPost, "http://x/", nil)
	require.NoError(t, err)

	n := SetSegmentedBody(req, [][]byte{head, payload})
	require.Equal(t, int64(len(head)+len(payload)), n)
	require.Equal(t, n, req.ContentLength)
	body, err := io.ReadAll(req.Body)
	require.NoError(t, err)
	require.Equal(t, append(append([]byte{}, head...), payload...), body)

	// GetBody lets the HTTP/2 transport replay the body after a GOAWAY.
	require.NotNil(t, req.GetBody)
	for i := range 2 {
		rc, err := req.GetBody()
		require.NoError(t, err)
		replay, err := io.ReadAll(rc)
		require.NoError(t, err)
		require.Equal(t, body, replay, "replay %d", i)
	}

	n = SetSegmentedBody(req, [][]byte{nil, {}})
	require.Zero(t, n)
	require.Equal(t, http.NoBody, req.Body)
	require.Zero(t, req.ContentLength)
	rc, err := req.GetBody()
	require.NoError(t, err)
	require.Equal(t, http.NoBody, rc)
}
