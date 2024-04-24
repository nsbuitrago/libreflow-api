mod io;

/// FCS file object
#[derive(Debug)]
pub struct File {
    inner: std::fs::File,
}
