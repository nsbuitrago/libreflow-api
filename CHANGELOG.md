# Change Log

## v0.0.1-alpha.1

### Changes

- Add `Gate` alias for `HashMap<String, EventData>`
- `Sample` now uses `Gate` to wrap `EventData`. By default,
  all event data is placed in a "root" gate. This allows for saving child gates
  directly in the sample object.
- Add serde serialization/deserialization to `Sample` struct.

## v0.0.1-alpha.0

### Features

- FCS header segment parsing
- FCS text segment parsing and validation
- FCS data segment parsing
