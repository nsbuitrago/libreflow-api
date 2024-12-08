pub mod io;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Metadata aliased as a hash-map with
#[derive(Debug, Deserialize, Serialize)]
pub type Metadata = HashMap<String, String>;

/// EventData aliased as a hash-map with parameter IDs as strings and their
/// event data as a vector of f64s.
#[derive(Debug, Deserialize, Serialize)]
pub type EventData = HashMap<String, Vec<f64>>;

/// FCS sample object containing metadata and event data.
#[derive(Debug, Deserialize, Serialize)]
pub struct Sample {
    metadata: Metadata,
    event_data: EventData,
}
