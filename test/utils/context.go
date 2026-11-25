/*
Copyright 2025 The llm-d Authors.

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

// Package utils contains utilities for testing
//
//revive:disable:var-naming
package utils

//revive:enable:var-naming

import (
	"context"
	"testing"

	"github.com/go-logr/logr/testr"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// NewTestContext creates a context derived from t.Context() with a testr logger.
// Because it derives from t.Context(), the returned context is canceled before
// t.Cleanup functions run.
func NewTestContext(t *testing.T) context.Context {
	t.Helper()

	logger := testr.New(t)
	return log.IntoContext(t.Context(), logger)
}
