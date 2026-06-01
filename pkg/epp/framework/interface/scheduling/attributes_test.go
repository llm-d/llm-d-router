/*
Copyright 2025 The Kubernetes Authors.

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

package scheduling

import (
	"strconv"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRequestAttributes_PutThenGet(t *testing.T) {
	r := &InferenceRequest{}

	r.PutAttribute("session", "abc")
	v, ok := r.GetAttribute("session")
	assert.True(t, ok)
	assert.Equal(t, "abc", v)

	_, ok = r.GetAttribute("missing")
	assert.False(t, ok)
}

func TestRequestAttributes_KeysAfterPuts(t *testing.T) {
	r := &InferenceRequest{}

	r.PutAttribute("a", 1)
	r.PutAttribute("b", "two")
	r.PutAttribute("a", 11) // overwrite

	assert.ElementsMatch(t, []string{"a", "b"}, r.AttributeKeys())
}

func TestReadRequestAttribute(t *testing.T) {
	r := &InferenceRequest{}
	r.PutAttribute("count", 42)
	r.PutAttribute("name", "alpha")

	count, ok := ReadRequestAttribute[int](r, "count")
	assert.True(t, ok)
	assert.Equal(t, 42, count)

	name, ok := ReadRequestAttribute[string](r, "name")
	assert.True(t, ok)
	assert.Equal(t, "alpha", name)

	missing, ok := ReadRequestAttribute[int](r, "absent")
	assert.False(t, ok)
	assert.Equal(t, 0, missing)

	mismatch, ok := ReadRequestAttribute[string](r, "count")
	assert.False(t, ok)
	assert.Equal(t, "", mismatch)
}

func TestRequestAttributes_ZeroValueRequestIsUsable(t *testing.T) {
	var r InferenceRequest

	r.PutAttribute("k", "v")
	v, ok := r.GetAttribute("k")
	assert.True(t, ok)
	assert.Equal(t, "v", v)
}

func TestRequestAttributes_ConcurrentAfterInit(t *testing.T) {
	r := &InferenceRequest{}
	r.PutAttribute("seed", 0) // ensure the store is allocated before concurrent writers start

	const writers = 8
	const writes = 200
	var wg sync.WaitGroup

	wg.Add(writers)
	for w := 0; w < writers; w++ {
		go func(id int) {
			defer wg.Done()
			for i := 0; i < writes; i++ {
				key := strconv.Itoa(id) + ":" + strconv.Itoa(i)
				r.PutAttribute(key, i)
				if v, ok := ReadRequestAttribute[int](r, key); !ok || v != i {
					t.Errorf("round-trip failed for %s: ok=%v v=%v", key, ok, v)
					return
				}
			}
		}(w)
	}
	wg.Wait()

	assert.Len(t, r.AttributeKeys(), writers*writes+1)
}
