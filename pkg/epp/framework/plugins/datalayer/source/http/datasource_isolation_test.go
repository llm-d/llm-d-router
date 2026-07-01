/*
Copyright 2026 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
you may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package http

import (
	"io"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHTTPDataSource_ClientIsolation(t *testing.T) {
	parser := func(r io.Reader) (any, error) { return struct{}{}, nil }

	// Create first HTTPS datasource, insecureSkipVerify = false
	ds1, err := NewHTTPDataSource[any]("https", "/metrics", false, "", "test-type", "ds1", parser)
	assert.NoError(t, err)

	// Create second HTTPS datasource, insecureSkipVerify = true
	ds2, err := NewHTTPDataSource[any]("https", "/metrics", true, "", "test-type", "ds2", parser)
	assert.NoError(t, err)

	// Verify ds1 uses isolated transport config
	cl1, ok := ds1.client.(*client)
	assert.True(t, ok)
	t1, ok := cl1.Transport.(*http.Transport)
	assert.True(t, ok)
	assert.NotNil(t, t1.TLSClientConfig)
	assert.False(t, t1.TLSClientConfig.InsecureSkipVerify)

	// Verify ds2 uses isolated transport config and does not pollute ds1
	cl2, ok := ds2.client.(*client)
	assert.True(t, ok)
	t2, ok := cl2.Transport.(*http.Transport)
	assert.True(t, ok)
	assert.NotNil(t, t2.TLSClientConfig)
	assert.True(t, t2.TLSClientConfig.InsecureSkipVerify)

	// Verify ds1 remains false (no configuration pollution)
	assert.False(t, t1.TLSClientConfig.InsecureSkipVerify)
}
