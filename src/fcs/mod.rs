pub mod io;

use std::collections::HashMap;

/// Metadata aliased as a hash-map with
pub type Metadata = HashMap<String, String>;

/// EventData aliased as a hash-map with parameter IDs as strings and their
/// event data as a vector of f64s.
pub type EventData = HashMap<String, Vec<f64>>;

/// FCS sample object containing metadata and event data.
pub struct Sample {
    metadata: Metadata,
    event_data: EventData,
}
