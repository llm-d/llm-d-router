package steps

import "bytes"

// All step init() functions register themselves via pipeline.Register.
// Import this package to trigger registration of all built-in steps.

func jsonReader(data []byte) *bytes.Reader {
	return bytes.NewReader(data)
}
