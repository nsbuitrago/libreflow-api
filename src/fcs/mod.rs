mod io;

use io::Metadata;
use std::collections::HashMap;

/// FCS Sample object
#[derive(Debug)]
pub struct Sample {
    metadata: Metadata,
    data: HashMap<String, Vec<f64>>,
}

/// FCS experiment object
pub struct Experiment {
    samples: Vec<Sample>,
}
