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

package proxy

// Routed-experts merging for P/D disaggregation.
//
// With --enable-return-routed-experts each response choice carries a
// "routed_experts" field: a base64-encoded NumPy .npy blob of shape
// (num_tokens-1, num_layers, num_experts_per_tok). Under P/D the decode replica
// pulls the prompt KV and never forwards the prompt, so its prompt-region rows
// are invalid; we splice the prefill replica's rows over them. Connector-agnostic,
// non-streaming only.

import (
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"strconv"
	"strings"
)

const requestFieldRoutedExperts = "routed_experts"

type npyHeader struct {
	descr      string
	shape      []int
	dataOffset int
}

// itemsize derives the element size in bytes from descr ("|u1" -> 1, "<u2" -> 2).
func (h npyHeader) itemsize() int {
	n := 0
	for _, c := range h.descr {
		if c >= '0' && c <= '9' {
			n = n*10 + int(c-'0')
		}
	}
	if n == 0 {
		return 1
	}
	return n
}

func (h npyHeader) rowBytes() int {
	n := 1
	for _, d := range h.shape[1:] {
		n *= d
	}
	return n * h.itemsize()
}

// parseNpyHeader parses a numpy v1.0 .npy header, assuming the well-formed
// payload vLLM emits (magic/version not re-checked, 2-byte header length).
func parseNpyHeader(buf []byte) npyHeader {
	dataOffset := 10 + int(binary.LittleEndian.Uint16(buf[8:10]))
	header := string(buf[10:dataOffset])

	var shape []int
	if open := strings.Index(header, "("); open >= 0 {
		if end := strings.Index(header[open:], ")"); end >= 0 {
			for _, part := range strings.Split(header[open+1:open+end], ",") {
				if part = strings.TrimSpace(part); part != "" {
					d, _ := strconv.Atoi(part)
					shape = append(shape, d)
				}
			}
		}
	}
	return npyHeader{descr: quoted(header, "descr"), shape: shape, dataOffset: dataOffset}
}

// quoted returns the single-quoted value for key (e.g. "descr" -> "<u2").
func quoted(header, key string) string {
	rest := header
	if i := strings.Index(rest, "'"+key+"':"); i >= 0 {
		rest = rest[i+len(key)+3:]
	} else {
		return ""
	}
	start := strings.Index(rest, "'")
	if start < 0 {
		return ""
	}
	rest = rest[start+1:]
	end := strings.Index(rest, "'")
	if end < 0 {
		return ""
	}
	return rest[:end]
}

// spliceRoutedExpertsNpy splices the prefill array's rows over the decode array's
// invalid prompt-region prefix. Inputs are base64 .npy blobs from a P/D pair
// (same dtype/trailing shape, prefill rows <= decode rows); the result reuses
// decode's header with its leading rows replaced by prefill's.
func spliceRoutedExpertsNpy(prefillB64, decodeB64 string) (string, error) {
	pBuf, err := base64.StdEncoding.DecodeString(prefillB64)
	if err != nil {
		return "", err
	}
	dBuf, err := base64.StdEncoding.DecodeString(decodeB64)
	if err != nil {
		return "", err
	}
	ph := parseNpyHeader(pBuf)
	dh := parseNpyHeader(dBuf)

	// Output keeps decode's header (shape unchanged); only the leading Lp rows
	// of payload come from prefill, the rest from decode.
	split := ph.shape[0] * dh.rowBytes()
	out := make([]byte, 0, len(dBuf))
	out = append(out, dBuf[:dh.dataOffset]...)
	out = append(out, pBuf[ph.dataOffset:ph.dataOffset+split]...)
	out = append(out, dBuf[dh.dataOffset+split:]...)
	return base64.StdEncoding.EncodeToString(out), nil
}

// extractRoutedExperts returns a map from choice index to that choice's
// base64-encoded routed_experts string, for any choice that carries one.
func extractRoutedExperts(response map[string]any) map[int]string {
	choices, ok := response["choices"].([]any)
	if !ok {
		return nil
	}
	out := map[int]string{}
	for i, c := range choices {
		choice, ok := c.(map[string]any)
		if !ok {
			continue
		}
		re, ok := choice[requestFieldRoutedExperts].(string)
		if !ok || re == "" {
			continue
		}
		idx := i
		if v, ok := intValue(choice["index"]); ok {
			idx = v
		}
		out[idx] = re
	}
	return out
}

// mergeRoutedExpertsJSON splices prefill routed_experts (keyed by choice index)
// into a decode response body. Returns the rewritten body and whether any choice
// was merged. On a per-choice splice error the choice is left unchanged.
func mergeRoutedExpertsJSON(prefillByIdx map[int]string, body []byte) ([]byte, bool) {
	if len(prefillByIdx) == 0 {
		return body, false
	}
	var response map[string]any
	if err := json.Unmarshal(body, &response); err != nil {
		return body, false
	}
	choices, ok := response["choices"].([]any)
	if !ok {
		return body, false
	}

	merged := false
	for i, c := range choices {
		choice, ok := c.(map[string]any)
		if !ok {
			continue
		}
		decodeRE, ok := choice[requestFieldRoutedExperts].(string)
		if !ok || decodeRE == "" {
			continue
		}
		idx := i
		if v, ok := intValue(choice["index"]); ok {
			idx = v
		}
		prefillRE, ok := prefillByIdx[idx]
		if !ok {
			continue
		}
		spliced, err := spliceRoutedExpertsNpy(prefillRE, decodeRE)
		if err != nil {
			continue
		}
		choice[requestFieldRoutedExperts] = spliced
		merged = true
	}
	if !merged {
		return body, false
	}
	updated, err := json.Marshal(response)
	if err != nil {
		return body, false
	}
	return updated, true
}
