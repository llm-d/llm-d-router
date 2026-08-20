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

package types

import (
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestQueueCapacityError(t *testing.T) {
	t.Parallel()

	t.Run("unwraps to the capacity sentinel", func(t *testing.T) {
		t.Parallel()
		err := fmt.Errorf("%w: %w", ErrRejected, &QueueCapacityError{RetryAfterHint: 5 * time.Second})
		assert.ErrorIs(t, err, ErrQueueAtCapacity, "sentinel matching must survive the typed wrapper")
		assert.ErrorIs(t, err, ErrRejected, "the outer rejection sentinel must remain matchable")
	})

	t.Run("carries the hint through the chain", func(t *testing.T) {
		t.Parallel()
		err := fmt.Errorf("%w: %w", ErrRejected, &QueueCapacityError{RetryAfterHint: 5 * time.Second})
		var capErr *QueueCapacityError
		require.ErrorAs(t, err, &capErr)
		assert.Equal(t, 5*time.Second, capErr.RetryAfterHint)
	})

	t.Run("message includes the projected wait only when set", func(t *testing.T) {
		t.Parallel()
		withHint := &QueueCapacityError{RetryAfterHint: 5 * time.Second}
		assert.Contains(t, withHint.Error(), ErrQueueAtCapacity.Error())
		assert.Contains(t, withHint.Error(), "5s")
		withoutHint := &QueueCapacityError{}
		assert.Equal(t, ErrQueueAtCapacity.Error(), withoutHint.Error())
	})
}
