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

// Package crossreplica holds what the cross-replica syncers share: the gob
// encoding of contributor values and the bounds of their configuration.
package crossreplica

import (
	"bytes"
	"encoding/gob"
	"errors"
	"reflect"
	"time"
)

// MinimumDuration is the shortest duration a syncer accepts in its
// configuration.
const MinimumDuration = time.Millisecond

// EncodeValue gob-encodes a contributor value.
func EncodeValue(value any) ([]byte, error) {
	if value == nil {
		return nil, errors.New("cannot encode nil value")
	}
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(value); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// DecodeValue decodes data into a new value of the prototype's type.
func DecodeValue(data []byte, prototype any) (any, error) {
	if prototype == nil {
		return nil, errors.New("cannot decode stored value without a type prototype")
	}
	target := reflect.New(reflect.TypeOf(prototype))
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(target.Interface()); err != nil {
		return nil, err
	}
	return target.Elem().Interface(), nil
}
