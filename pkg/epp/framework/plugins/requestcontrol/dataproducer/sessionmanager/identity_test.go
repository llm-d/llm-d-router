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
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestIdentityGoldenVector(t *testing.T) {
	t.Parallel()
	key := make([]byte, 32)
	for i := range key {
		key[i] = byte(i)
	}
	tag, scope := deriveIdentity(key, "payments-prod-eu1", "codex-17")
	assert.Equal(t, "vFtm8tPY1rcwryf9y2-BnCo6oP-8omPfLKroGwDhYPI", tag)
	assert.Equal(t, "qwouieMAevJn-7pLwGLqaIWQoBfGHZr1Uw78mf0aZNk", scope)
}

func TestIdentityScopeInputsSeparateTags(t *testing.T) {
	t.Parallel()
	key := make([]byte, 32)
	tag, scope := deriveIdentity(key, "deployment-a", "alias")
	otherTag, otherScope := deriveIdentity(key, "deployment-b", "alias")
	assert.NotEqual(t, tag, otherTag)
	assert.NotEqual(t, scope, otherScope)

	otherKey := append([]byte(nil), key...)
	otherKey[0] = 1
	rotatedTag, rotatedScope := deriveIdentity(otherKey, "deployment-a", "alias")
	assert.NotEqual(t, tag, rotatedTag)
	assert.NotEqual(t, scope, rotatedScope)
}
