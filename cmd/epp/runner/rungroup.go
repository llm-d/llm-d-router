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

package runner

import (
	"context"
	"fmt"

	"golang.org/x/sync/errgroup"
)

// RunnableGroup collects named goroutines and starts them together under a
// shared context. The first non-nil error cancels the group; Run returns that
// error wrapped with the runnable name.
type RunnableGroup interface {
	Add(name string, fn func(ctx context.Context) error)
	Run(ctx context.Context) error
}

func newRunnableGroup() RunnableGroup {
	return &groupRunner{}
}

type namedFn struct {
	name string
	fn   func(ctx context.Context) error
}

type groupRunner struct {
	fns []namedFn
}

func (g *groupRunner) Add(name string, fn func(ctx context.Context) error) {
	g.fns = append(g.fns, namedFn{name: name, fn: fn})
}

func (g *groupRunner) Run(ctx context.Context) error {
	eg, ectx := errgroup.WithContext(ctx)
	for _, nf := range g.fns {
		eg.Go(func() error {
			if err := nf.fn(ectx); err != nil {
				return fmt.Errorf("%s: %w", nf.name, err)
			}
			return nil
		})
	}
	return eg.Wait()
}
