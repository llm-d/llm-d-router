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

package kvblock

import (
	"io"
	"testing"

	"github.com/redis/go-redis/v9"
	"github.com/stretchr/testify/assert"
)

// fakeRedisError implements redis.Error so HasErrorPrefix recognizes it as a
// server-reported error, without depending on go-redis's unexported
// proto.RedisError (the type real WRONGTYPE replies arrive as).
type fakeRedisError string

func (e fakeRedisError) Error() string { return string(e) }
func (fakeRedisError) RedisError()     {}

var _ redis.Error = fakeRedisError("")

// wrongTypeErr mirrors the error go-redis returns for a Redis WRONGTYPE
// reply: a message carrying the prefix HasErrorPrefix checks for.
var wrongTypeErr = fakeRedisError("WRONGTYPE Operation against a key holding the wrong kind of value")

func stringSliceCmd(err error) *redis.StringSliceCmd {
	return redis.NewStringSliceResult(nil, err)
}

// TestIsPipelineFailure covers the classification isPipelineFailure makes:
// a per-key WRONGTYPE conflict (even as the only or every result) is not a
// pipeline failure, but any other error present anywhere in the batch is --
// including a partial failure where only some commands carry it, which a
// real connection drop mid-stream can produce (some responses already
// parsed before the break).
func TestIsPipelineFailure(t *testing.T) {
	tests := []struct {
		name    string
		results []*redis.StringSliceCmd
		want    bool
	}{
		{"empty batch", nil, false},
		{"single key, no error", []*redis.StringSliceCmd{stringSliceCmd(nil)}, false},
		{"single key, WRONGTYPE", []*redis.StringSliceCmd{stringSliceCmd(wrongTypeErr)}, false},
		{"all keys WRONGTYPE", []*redis.StringSliceCmd{stringSliceCmd(wrongTypeErr), stringSliceCmd(wrongTypeErr)}, false},
		{"one WRONGTYPE, one clean", []*redis.StringSliceCmd{stringSliceCmd(wrongTypeErr), stringSliceCmd(nil)}, false},
		{"single key, transport error", []*redis.StringSliceCmd{stringSliceCmd(io.EOF)}, true},
		{"all keys, transport error", []*redis.StringSliceCmd{stringSliceCmd(io.EOF), stringSliceCmd(io.EOF)}, true},
		{"partial failure: clean then transport error", []*redis.StringSliceCmd{stringSliceCmd(nil), stringSliceCmd(io.EOF)}, true},
		{"partial failure: WRONGTYPE then transport error", []*redis.StringSliceCmd{stringSliceCmd(wrongTypeErr), stringSliceCmd(io.EOF)}, true},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, isPipelineFailure(tc.results))
		})
	}
}
