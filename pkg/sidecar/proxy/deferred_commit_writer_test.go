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
	"io"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/llm-d/llm-d-router/pkg/common/grpc/grpcframe"
	reqcommon "github.com/llm-d/llm-d-router/pkg/common/request"
)

func TestDeferredCommitWriterConcurrentCommitAndTrailerWrite(t *testing.T) {
	for range 100 {
		dcw := newDeferredCommitWriter()
		dcw.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
		dcw.WriteHeader(http.StatusOK)
		_, err := dcw.Write([]byte("body"))
		require.NoError(t, err)

		rec := httptest.NewRecorder()
		start := make(chan struct{})
		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			<-start
			dcw.Header().Add(http.TrailerPrefix+grpcframe.HeaderStatus, grpcframe.StatusOK)
			dcw.Header().Add(http.TrailerPrefix+grpcframe.HeaderMessage, "ok")
		}()

		close(start)
		require.True(t, dcw.commit(newVLLMCachedTokensResponseWriter(rec, 4)))
		wg.Wait()
		dcw.flushTrailers()

		res := rec.Result()
		assert.Equal(t, http.StatusOK, res.StatusCode)
		assert.Equal(t, grpcframe.MediaType, res.Header.Get(reqcommon.HeaderContentType))
		assert.Equal(t, grpcframe.StatusOK, res.Trailer.Get(grpcframe.HeaderStatus))
		assert.Equal(t, "ok", res.Trailer.Get(grpcframe.HeaderMessage))
		_ = res.Body.Close()
	}
}

func TestDeferredCommitWriterTrailersRelayedAcrossCommitTimings(t *testing.T) {
	t.Run("unannounced and announced trailers buffered before commit", func(t *testing.T) {
		dcw := newDeferredCommitWriter()
		dcw.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
		dcw.Header().Add("Trailer", "X-Announced")
		dcw.WriteHeader(http.StatusOK)
		_, err := dcw.Write([]byte("payload"))
		require.NoError(t, err)

		dcw.Header().Add(http.TrailerPrefix+grpcframe.HeaderStatus, grpcframe.StatusOK)
		dcw.Header().Add("X-Announced", "done")

		rec := httptest.NewRecorder()
		require.True(t, dcw.commit(rec))
		dcw.flushTrailers()

		res := rec.Result()
		defer res.Body.Close()
		body, err := io.ReadAll(res.Body)
		require.NoError(t, err)
		assert.Equal(t, "payload", string(body))
		assert.Equal(t, grpcframe.MediaType, res.Header.Get(reqcommon.HeaderContentType))
		assert.Empty(t, res.Header.Get("X-Announced"))
		assert.Equal(t, "done", res.Trailer.Get("X-Announced"))
		assert.Equal(t, grpcframe.StatusOK, res.Trailer.Get(grpcframe.HeaderStatus))
	})

	t.Run("trailers split before and after commit keep order", func(t *testing.T) {
		dcw := newDeferredCommitWriter()
		dcw.WriteHeader(http.StatusOK)
		dcw.Header().Add(http.TrailerPrefix+"X-Trace", "before")

		rec := httptest.NewRecorder()
		require.True(t, dcw.commit(rec))
		dcw.Header().Add(http.TrailerPrefix+"X-Trace", "after")
		dcw.flushTrailers()

		res := rec.Result()
		defer res.Body.Close()
		assert.Equal(t, []string{"before", "after"}, res.Trailer.Values("X-Trace"))
	})

	t.Run("gRPC error trailer buffered before commit marks writer failed", func(t *testing.T) {
		dcw := newDeferredCommitWriter()
		dcw.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
		dcw.WriteHeader(http.StatusOK)
		dcw.Header().Add(http.TrailerPrefix+grpcframe.HeaderStatus, "13")

		assert.True(t, dcw.failed())

		rec := httptest.NewRecorder()
		require.True(t, dcw.commit(rec))
		dcw.flushTrailers()

		assert.True(t, dcw.failed())
		res := rec.Result()
		defer res.Body.Close()
		assert.Equal(t, "13", res.Trailer.Get(grpcframe.HeaderStatus))
	})

	t.Run("abort drops buffered trailers without touching dst", func(t *testing.T) {
		dcw := newDeferredCommitWriter()
		dcw.WriteHeader(http.StatusOK)
		dcw.Header().Add(http.TrailerPrefix+grpcframe.HeaderStatus, "13")
		dcw.abort()
		dcw.flushTrailers()

		assert.False(t, dcw.responseStarted())
		assert.False(t, dcw.commit(httptest.NewRecorder()))
	})

	t.Run("informational 1xx status does not freeze initial headers", func(t *testing.T) {
		dcw := newDeferredCommitWriter()
		h := dcw.Header()
		h.Set("X-Hint", "1")
		dcw.WriteHeader(http.StatusEarlyHints)
		clear(h)
		dcw.Header().Set(reqcommon.HeaderContentType, grpcframe.MediaType)
		dcw.WriteHeader(http.StatusOK)

		rec := httptest.NewRecorder()
		require.True(t, dcw.commit(rec))
		dcw.flushTrailers()

		res := rec.Result()
		defer res.Body.Close()
		assert.Equal(t, http.StatusOK, res.StatusCode)
		assert.Empty(t, res.Header.Get("X-Hint"))
		assert.Equal(t, grpcframe.MediaType, res.Header.Get(reqcommon.HeaderContentType))
	})
}
