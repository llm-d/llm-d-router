# Stable plugin configurations

This directory holds the frozen configuration surface of plugins registered with
`plugin.StabilityStable`. Each file is a complete `EndpointPickerConfig` that
exercises one stable plugin and every parameter it accepts.

## These files must not be changed

A plugin is promoted to Stable on the promise that a configuration valid today
stays valid for the whole major version. These files *are* that promise, written
down. `TestStablePluginConfigs` loads every one of them on each run, so a change
to a stable plugin's parameters, defaults, or validation that would break an
existing deployment fails CI here first.

That only works if the files themselves never change. Concretely:

- **Do not** edit a file in `v1/` to make a failing test pass. A failure means the
  plugin changed incompatibly; fix the plugin, or take it back out of Stable.
- **Do not** rename, move, or delete a file for the lifetime of the major version.
- **Do** add a new file when a plugin is newly promoted to Stable.
- **Do** add a parameter to an existing file only when that parameter is itself
  new and optional, so the file continues to describe a config that a user could
  already have written.

Changes that cannot be expressed that way belong in the next major version's
directory (`v2/`), not in an edit to `v1/`.

## Layout

```
test/testdata/plugins/stable/
  README.md          <- this file
  v1/                <- the frozen contract for the v1 line
    <plugin-type>.yaml
```

One file per plugin, named after the plugin type it covers, so a reader can find
the frozen config for a plugin without opening anything.

The directory is versioned by *major* version because that is the unit the
stability guarantee is scoped to. 0.x carries no stability promise, so configs
land in `v1/` now and stay put when v1 is cut, rather than moving at the release
and breaking the "these paths never move" property this directory depends on.

## Adding a newly promoted plugin

1. Flip the plugin's registration in `cmd/epp/runner/runner.go` from
   `plugin.StabilityBeta` to `plugin.StabilityStable`.
2. Add `v1/<plugin-type>.yaml` here — a minimal config that instantiates the
   plugin and sets every parameter it accepts, with values a real deployment
   would use.
3. Run `go test ./cmd/epp/runner/ -run TestStablePluginConfigs`.

No test registration is needed; the test discovers files by globbing this
directory.
