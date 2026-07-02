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

package pipeline

import (
	"context"
	"errors"
	"testing"
)

type mockStep struct {
	name string
	fn   func(ctx context.Context, rc *RequestContext) error
}

func (m *mockStep) Name() string { return m.name }
func (m *mockStep) Execute(ctx context.Context, rc *RequestContext) error {
	return m.fn(ctx, rc)
}

func TestPipeline_ExecutesStepsInOrder(t *testing.T) {
	var order []string
	steps := []Step{
		&mockStep{name: "a", fn: func(_ context.Context, _ *RequestContext) error {
			order = append(order, "a")
			return nil
		}},
		&mockStep{name: "b", fn: func(_ context.Context, _ *RequestContext) error {
			order = append(order, "b")
			return nil
		}},
		&mockStep{name: "c", fn: func(_ context.Context, _ *RequestContext) error {
			order = append(order, "c")
			return nil
		}},
	}

	p := New(steps)
	err := p.Execute(context.Background(), &RequestContext{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(order) != 3 || order[0] != "a" || order[1] != "b" || order[2] != "c" {
		t.Fatalf("unexpected execution order: %v", order)
	}
}

func TestPipeline_AbortsOnError(t *testing.T) {
	executed := map[string]bool{}
	steps := []Step{
		&mockStep{name: "a", fn: func(_ context.Context, _ *RequestContext) error {
			executed["a"] = true
			return errors.New("step a failed")
		}},
		&mockStep{name: "b", fn: func(_ context.Context, _ *RequestContext) error {
			executed["b"] = true
			return nil
		}},
	}

	p := New(steps)
	err := p.Execute(context.Background(), &RequestContext{})
	if err == nil {
		t.Fatal("expected error")
	}
	if !executed["a"] {
		t.Fatal("step a should have executed")
	}
	if executed["b"] {
		t.Fatal("step b should NOT have executed")
	}
}

func TestPipeline_StopsOnErrPipelineDone(t *testing.T) {
	executed := map[string]bool{}
	steps := []Step{
		&mockStep{name: "a", fn: func(_ context.Context, _ *RequestContext) error {
			executed["a"] = true
			return ErrPipelineDone
		}},
		&mockStep{name: "b", fn: func(_ context.Context, _ *RequestContext) error {
			executed["b"] = true
			return nil
		}},
	}

	p := New(steps)
	err := p.Execute(context.Background(), &RequestContext{})
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if !executed["a"] {
		t.Fatal("step a should have executed")
	}
	if executed["b"] {
		t.Fatal("step b should NOT have executed after ErrPipelineDone")
	}
}

func TestPipeline_RespectsContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	steps := []Step{
		&mockStep{name: "a", fn: func(_ context.Context, _ *RequestContext) error {
			t.Fatal("should not execute")
			return nil
		}},
	}

	p := New(steps)
	err := p.Execute(ctx, &RequestContext{})
	if err == nil {
		t.Fatal("expected cancellation error")
	}
}
