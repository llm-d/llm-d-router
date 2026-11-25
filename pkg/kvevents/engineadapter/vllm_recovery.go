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

package engineadapter

import (
	"fmt"
	"slices"

	"github.com/llm-d/llm-d-router/pkg/kvevents"
	"github.com/vmihailenco/msgpack/v5"
	"github.com/vmihailenco/msgpack/v5/msgpcode"
)

func isRecoveryUnitTag(tag string, schema *eventWireSchema) bool {
	return schema.recoveryUnits && (tag == string(kvevents.EventTypeRecoveryUnitStored) || tag == string(kvevents.EventTypeRecoveryUnitRemoved))
}

type recoveryFields struct {
	hash      *kvevents.RecoveryUnitHash
	numTokens *int
	frontier  *bool
	cause     *string
	tier      *string
	numBlocks *int
}

var recoveryIdentityFields = []eventFieldDecoder{
	{"unit_hash", decodeUnitHash},
	{"num_tokens", func(d *msgpack.Decoder, f *typedEventFields) error {
		n, err := decodeInt(d)
		if err == nil && n < 0 {
			return fmt.Errorf("negative num_tokens: %d", n)
		}
		f.recovery.numTokens = &n
		return err
	}},
	{"frontier", func(d *msgpack.Decoder, f *typedEventFields) error {
		code, err := d.PeekCode()
		if err != nil {
			return err
		}
		if code != msgpcode.True && code != msgpcode.False {
			return fmt.Errorf("frontier requires a boolean")
		}
		v, err := d.DecodeBool()
		f.recovery.frontier = &v
		return err
	}},
}

var recoveryStoredFields = append(slices.Clip(recoveryIdentityFields), eventFieldDecoder{"session_id", decodeSessionID})
var recoveryRemovedFields = append(slices.Clip(recoveryIdentityFields),
	eventFieldDecoder{"cause", func(d *msgpack.Decoder, f *typedEventFields) (err error) {
		f.recovery.cause, err = decodeOptionalString(d)
		return err
	}},
	eventFieldDecoder{"tier", func(d *msgpack.Decoder, f *typedEventFields) (err error) {
		f.recovery.tier, err = decodeOptionalString(d)
		return err
	}},
	eventFieldDecoder{"num_blocks", func(d *msgpack.Decoder, f *typedEventFields) (err error) {
		f.recovery.numBlocks, err = decodeOptionalInt(d)
		return err
	}},
)

func decodeUnitHash(d *msgpack.Decoder, f *typedEventFields) error {
	code, err := d.PeekCode()
	if err != nil {
		return err
	}
	key := &kvevents.RecoveryUnitHash{}
	switch {
	case msgpcode.IsBin(code):
		size, err := d.DecodeBytesLen()
		if err != nil {
			return err
		}
		if size < 1 || size > maxDecodeHashBytes {
			return fmt.Errorf("invalid unit_hash length: %d", size)
		}
		b := make([]byte, size)
		if err := d.ReadFull(b); err != nil {
			return err
		}
		key.Bytes = string(b)
	case code <= msgpcode.PosFixedNumHigh || code == msgpcode.Uint8 || code == msgpcode.Uint16 || code == msgpcode.Uint32 || code == msgpcode.Uint64:
		key.Integer, err = d.DecodeUint64()
	default:
		var value int
		value, err = decodeInt(d)
		if err != nil {
			return err
		}
		if value < 0 {
			return fmt.Errorf("negative unit_hash: %d", value)
		}
		key.Integer = uint64(value)
	}
	f.recovery.hash = key
	return err
}

func (f *recoveryFields) event(tag string, session *string) (kvevents.GenericEvent, error) {
	if f.hash == nil || f.numTokens == nil || f.frontier == nil {
		return nil, fmt.Errorf("%s requires unit_hash, num_tokens and frontier", tag)
	}
	unit := kvevents.RecoveryUnit{UnitHash: *f.hash, NumTokens: *f.numTokens, Frontier: *f.frontier}
	if tag == string(kvevents.EventTypeRecoveryUnitStored) {
		return &kvevents.RecoveryUnitStoredEvent{RecoveryUnit: unit, SessionID: session}, nil
	}
	if f.cause == nil {
		return nil, fmt.Errorf("%s requires cause", tag)
	}
	return &kvevents.RecoveryUnitRemovedEvent{RecoveryUnit: unit, Cause: *f.cause, Tier: f.tier, NumBlocks: f.numBlocks}, nil
}
