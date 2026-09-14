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

package sessionmanager

import (
	"encoding/base64"
	"encoding/binary"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestStampCounterRemainsExhausted(t *testing.T) {
	generator := &stampGenerator{}
	generator.counter.Store(^uint64(0) - 1)

	_, err := generator.next()
	require.NoError(t, err)
	_, err = generator.next()
	require.ErrorContains(t, err, "exhausted")
	_, err = generator.next()
	require.ErrorContains(t, err, "exhausted")
}

func TestStampGeneratorRestartChangesNonce(t *testing.T) {
	first, err := newStampGenerator()
	require.NoError(t, err)
	second, err := newStampGenerator()
	require.NoError(t, err)

	firstStamp, err := first.next()
	require.NoError(t, err)
	secondStamp, err := second.next()
	require.NoError(t, err)
	firstRaw, err := base64.RawURLEncoding.DecodeString(firstStamp)
	require.NoError(t, err)
	secondRaw, err := base64.RawURLEncoding.DecodeString(secondStamp)
	require.NoError(t, err)

	assert.NotEqual(t, firstRaw[:16], secondRaw[:16])
	assert.Equal(t, uint64(1), binary.BigEndian.Uint64(firstRaw[16:]))
	assert.Equal(t, uint64(1), binary.BigEndian.Uint64(secondRaw[16:]))
}
