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
	"crypto/rand"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"sync/atomic"
)

type stampGenerator struct {
	nonce   [16]byte
	counter atomic.Uint64
}

func newStampGenerator() (*stampGenerator, error) {
	generator := &stampGenerator{}
	if _, err := rand.Read(generator.nonce[:]); err != nil {
		return nil, err
	}
	return generator, nil
}

func (g *stampGenerator) next() (string, error) {
	var counter uint64
	for {
		current := g.counter.Load()
		if current == ^uint64(0) {
			return "", errors.New("request stamp counter exhausted")
		}
		counter = current + 1
		if g.counter.CompareAndSwap(current, counter) {
			break
		}
	}
	var raw [24]byte
	copy(raw[:16], g.nonce[:])
	binary.BigEndian.PutUint64(raw[16:], counter)
	return base64.RawURLEncoding.EncodeToString(raw[:]), nil
}
