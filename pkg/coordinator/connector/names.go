// Package connector holds shared name constants used by the kv and ec
// subpackages. Connector implementations live in pkg/connector/kv and
// pkg/connector/ec.
package connector

// Connector name strings. The same string values are reused for both KV and
// EC connector selection, so a config can pair, e.g., nixlv2-KV with
// shared_storage-EC by passing the same constant in different fields.
const (
	NameNIXLv2        = "nixlv2"
	NameSharedStorage = "shared_storage"
	NameSGLang        = "sglang"
)
