/*
Copyright 2026 The Kubernetes Authors.

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

package statestore

// AccessMode controls how a shared state capability is accessed. Each of the
// three capabilities (inflight, prefix, flow control) can be configured with
// an independent access mode, enabling per-capability rollout: AccessModeLocal
// forces classic-equivalent behavior for that capability even when a remote
// client is configured, so a deployment can enable the State API for one
// capability at a time.
type AccessMode string

const (
	// AccessModeLocal means no remote access; the capability is equivalent to
	// classic. Used for phased rollout. The zero value is treated as Local.
	AccessModeLocal AccessMode = "Local"

	// AccessModeFailOpen prefers a remote read (global view) and falls back to
	// the local shadow on failure. Used by inflight and prefix capabilities.
	AccessModeFailOpen AccessMode = "FailOpen"

	// AccessModeLocalFallback is flow-control only: the local quota always
	// runs, and the local quota takes over when the remote fails (N x localMax).
	AccessModeLocalFallback AccessMode = "LocalFallback"
)

// AccessModeConfig holds the per-capability access mode configuration, set via
// the --state-access-mode-inflight, --state-access-mode-prefix, and
// --state-access-mode-flowcontrol flags.
type AccessModeConfig struct {
	// Inflight controls how inflight counters are accessed.
	Inflight AccessMode
	// Prefix controls how the prefix cache index is accessed.
	Prefix AccessMode
	// FlowControl controls how flow control quotas are accessed.
	FlowControl AccessMode
}

// DefaultAccessModeConfig returns the access mode configuration used when none
// is explicitly set. All capabilities default to AccessModeLocal so that
// behavior is equivalent to classic and existing deployments require zero
// changes.
func DefaultAccessModeConfig() AccessModeConfig {
	return AccessModeConfig{
		Inflight:    AccessModeLocal,
		Prefix:      AccessModeLocal,
		FlowControl: AccessModeLocal,
	}
}
