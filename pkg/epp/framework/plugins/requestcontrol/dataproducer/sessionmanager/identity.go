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
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
)

const (
	derivationVersion = "llm-d/session-tag/v1"
	scopeDomain       = "llm-d/session-scope/v1"
)

func deriveIdentity(key []byte, deploymentID, alias string) (tag, scope string) {
	tag = digest(key, derivationVersion, deploymentID, alias)
	scope = digest(key, scopeDomain, derivationVersion, deploymentID)
	return tag, scope
}

func digest(key []byte, fields ...string) string {
	mac := hmac.New(sha256.New, key)
	for _, field := range fields {
		raw := []byte(field)
		var length [8]byte
		binary.BigEndian.PutUint64(length[:], uint64(len(raw)))
		_, _ = mac.Write(length[:])
		_, _ = mac.Write(raw)
	}
	return base64.RawURLEncoding.EncodeToString(mac.Sum(nil))
}
