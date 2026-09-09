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

package tokenizer

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"unicode/utf16"
	"unicode/utf8"
)

// pythonDumps re-serializes raw JSON the way CPython's json.dumps does with
// default settings (separators ", " and ": ", ensure_ascii=True), preserving
// object key order and number literals. vLLM embeds json.dumps(tool input)
// into the rendered prompt, so tool-call arguments must match byte-for-byte
// for prefix-cache hits.
func pythonDumps(raw json.RawMessage) (string, error) {
	var sb strings.Builder
	if err := dumpValue(&sb, bytes.TrimSpace(raw)); err != nil {
		return "", err
	}
	return sb.String(), nil
}

func dumpValue(sb *strings.Builder, raw []byte) error {
	switch {
	case len(raw) == 0:
		return errors.New("pythonDumps: empty JSON value")
	case raw[0] == '{':
		return dumpDelimited(sb, raw, '{')
	case raw[0] == '[':
		return dumpDelimited(sb, raw, '[')
	case raw[0] == '"':
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return fmt.Errorf("pythonDumps: decode string: %w", err)
		}
		writeJSONString(sb, s)
		return nil
	default:
		// Numbers, booleans, and null pass through as their wire literal.
		var n json.Number
		if string(raw) != "null" && string(raw) != "true" && string(raw) != "false" && json.Unmarshal(raw, &n) != nil {
			return fmt.Errorf("pythonDumps: invalid value %s", raw)
		}
		sb.Write(raw)
		return nil
	}
}

// dumpDelimited re-serializes an object or array, keyed on the opening
// delimiter.
func dumpDelimited(sb *strings.Builder, raw []byte, open byte) error {
	dec := json.NewDecoder(bytes.NewReader(raw))
	if _, err := dec.Token(); err != nil { // consume the opening delimiter
		return fmt.Errorf("pythonDumps: decode start: %w", err)
	}
	closer, sep := '}', ": "
	if open == '[' {
		closer, sep = ']', ", "
	}
	sb.WriteByte(open)
	first := true
	for dec.More() {
		if !first {
			sb.WriteString(", ")
		}
		first = false
		if open == '{' {
			tok, err := dec.Token()
			if err != nil {
				return fmt.Errorf("pythonDumps: decode object key: %w", err)
			}
			key, ok := tok.(string)
			if !ok {
				return fmt.Errorf("pythonDumps: unexpected object key %v", tok)
			}
			writeJSONString(sb, key)
			sb.WriteString(sep)
		}
		var val json.RawMessage
		if err := dec.Decode(&val); err != nil {
			return fmt.Errorf("pythonDumps: decode value: %w", err)
		}
		if err := dumpValue(sb, bytes.TrimSpace(val)); err != nil {
			return err
		}
	}
	sb.WriteRune(closer)
	return nil
}

// writeJSONString writes an already-decoded string as a CPython JSON string
// literal: short escapes for the classic control characters, \u00xx for the
// rest of the control range, and \uxxxx (surrogate pairs above the BMP) for
// all non-ASCII runes, matching ensure_ascii=True.
func writeJSONString(sb *strings.Builder, s string) {
	sb.WriteByte('"')
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		switch {
		case r == '"':
			sb.WriteString(`\"`)
		case r == '\\':
			sb.WriteString(`\\`)
		case r == '\b':
			sb.WriteString(`\b`)
		case r == '\f':
			sb.WriteString(`\f`)
		case r == '\n':
			sb.WriteString(`\n`)
		case r == '\r':
			sb.WriteString(`\r`)
		case r == '\t':
			sb.WriteString(`\t`)
		case r < 0x20 || r == 0x7f:
			fmt.Fprintf(sb, `\u%04x`, r)
		case r < utf8.RuneSelf:
			sb.WriteByte(byte(r))
		case r > 0xFFFF:
			r1, r2 := utf16.EncodeRune(r)
			fmt.Fprintf(sb, `\u%04x\u%04x`, r1, r2)
		default:
			fmt.Fprintf(sb, `\u%04x`, r)
		}
		i += size
	}
	sb.WriteByte('"')
}
