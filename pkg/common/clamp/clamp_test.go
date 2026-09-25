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

package clamp_test

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/llm-d/llm-d-router/pkg/common/clamp"
)

func TestUint64(t *testing.T) {
	t.Parallel()

	assert.Equal(t, uint64(0), clamp.Uint64(0))
	assert.Equal(t, uint64(42), clamp.Uint64(42))
	assert.Equal(t, uint64(0), clamp.Uint64(-1), "a negative value must clamp to 0, not wrap")
	assert.Equal(t, uint64(42), clamp.Uint64(int32(42)))
	assert.Equal(t, uint64(42), clamp.Uint64(int64(42)))
}

func TestInt64(t *testing.T) {
	t.Parallel()

	assert.Equal(t, int64(0), clamp.Int64(0))
	assert.Equal(t, int64(42), clamp.Int64(42))
	assert.Equal(t, int64(math.MaxInt64), clamp.Int64(math.MaxInt64))
	assert.Equal(t, int64(math.MaxInt64), clamp.Int64(math.MaxUint64),
		"a value beyond the signed range must clamp to math.MaxInt64, not wrap")
}

func TestInt(t *testing.T) {
	t.Parallel()

	assert.Equal(t, 0, clamp.Int(0))
	assert.Equal(t, 42, clamp.Int(42))
	assert.Equal(t, math.MaxInt, clamp.Int(uint64(math.MaxInt)))
	assert.Equal(t, math.MaxInt, clamp.Int(math.MaxUint64),
		"a value beyond the signed range must clamp to math.MaxInt, not wrap")
}

func TestUint32(t *testing.T) {
	t.Parallel()

	assert.Equal(t, uint32(0), clamp.Uint32(0))
	assert.Equal(t, uint32(42), clamp.Uint32(42))
	assert.Equal(t, uint32(0), clamp.Uint32(-1), "a negative value must clamp to 0, not wrap")
	assert.Equal(t, uint32(math.MaxUint32), clamp.Uint32(int64(math.MaxUint32)))
	assert.Equal(t, uint32(math.MaxUint32), clamp.Uint32(int64(math.MaxUint32)+1),
		"a value beyond the unsigned 32-bit range must clamp to math.MaxUint32, not wrap")
}
