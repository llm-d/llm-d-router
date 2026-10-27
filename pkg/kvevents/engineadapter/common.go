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
	"math"
	"strings"

	"github.com/vmihailenco/msgpack/v5"
	"github.com/vmihailenco/msgpack/v5/msgpcode"
)

const (
	// Common event type tags shared across engine adapters.
	eventTagBlockStored      = "BlockStored"
	eventTagBlockRemoved     = "BlockRemoved"
	eventTagAllBlocksCleared = "AllBlocksCleared"
)

// parseTopic extracts pod ID and model name from the topic format "kv@<pod-id>@<model-name>".
func parseTopic(topic string) (string, string) {
	topicParts := strings.Split(topic, "@")
	if len(topicParts) == 3 {
		return topicParts[1], topicParts[2]
	}
	return topic, ""
}

func decodeTokenID(dec *msgpack.Decoder) (uint32, error) {
	code, err := dec.PeekCode()
	if err != nil {
		return 0, err
	}
	if !isIntegerCode(code) {
		return 0, fmt.Errorf("unsupported numeric type: MessagePack code %#x", code)
	}
	var value uint64
	if code >= msgpcode.NegFixedNumLow || code == msgpcode.Int8 || code == msgpcode.Int16 || code == msgpcode.Int32 || code == msgpcode.Int64 {
		signed, decodeErr := dec.DecodeInt64()
		err = decodeErr
		if signed > 0 {
			value = uint64(signed)
		}
	} else {
		value, err = dec.DecodeUint64()
	}
	if err != nil {
		return 0, err
	}
	return uint32(min(value, uint64(math.MaxUint32))), nil
}

func isIntegerCode(code byte) bool {
	return msgpcode.IsFixedNum(code) || code == msgpcode.Uint8 || code == msgpcode.Uint16 ||
		code == msgpcode.Uint32 || code == msgpcode.Uint64 || code == msgpcode.Int8 ||
		code == msgpcode.Int16 || code == msgpcode.Int32 || code == msgpcode.Int64
}
