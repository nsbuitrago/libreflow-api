use std::path::Path;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use thiserror::Error;
use atoi::atoi;
use std::collections::HashMap;

const VALID_FCS_VERSIONS: [&[u8; 6]; 2] = [b"FCS3.0", b"FCS3.1"];

/// Number of offsets to read from the FCS header. We set this to 2 since we only care about the
/// txt header segment. The offsets for the data and analysis segments are duplicated in the text
/// segment and we would have to check the text segment anyway. 
const N_FCS_OFFSETS: usize = 2;

#[derive(Debug, Error)]
pub enum FCSError {
    #[error("IO Error: {0}")]
    IOError(#[from] io::Error),
    #[error("Invalid FCS version. Found: `{0}`")]
    InvalidVersion(String),
    #[error("Invalid FCS header. File may be corrupted on not an FCS file.")]
    InvalidHeader,
    #[error("Invalid FCS metadata. Unable to parse ascii metadata.")]
    InvlidMetadata,
}

/// Read FCS file metadata and event data into Sample
///
/// This function provides parsing of FCS files into Sample structs.
///
/// # Errors
///
/// This function will return an error if `path` does not exist or is not a valid FCS file.
///
/// If the file is a valid FCS file, but the metadata or event data cannot be parsed, an error
/// will be returned.
///
/// # Examples
///
/// ```no_run
/// use libreflow_api::io::{read_fcs, FCSError};
///
/// fn main() -> Result<(), FCSError> {
///    let sample = read_fcs("./path/to/fcs_file.fcs")?;
///    Ok(())
///    }
/// ```
pub fn read_fcs<P: AsRef<Path>>(path: P) -> Result<HashMap<String, String>, FCSError> {
    let fcs_file = File::open(path)?;
    let mut reader = BufReader::new(fcs_file);
    let txt_offsets = parse_fcs_header(&mut reader)?;
    let metadata = parse_fcs_metadata(&mut reader, &txt_offsets)?;
    
    Ok(metadata)
}

/// Parse FCS header and return offsets to txt, data, and analysis segments
fn parse_fcs_header(reader: &mut BufReader<File>) -> Result<[u64; N_FCS_OFFSETS], FCSError> {

    let mut fcs_version = [0u8; 6];
    reader.read_exact(&mut fcs_version)?;

    if !VALID_FCS_VERSIONS.contains(&&fcs_version) {
        return Err(FCSError::InvalidVersion(
            String::from_utf8_lossy(&fcs_version).to_string(),
        ));
    } 

    reader.seek(SeekFrom::Current(4))?;
    let mut header_offsets = [0u64; N_FCS_OFFSETS];

    for i in 0..N_FCS_OFFSETS {
        let mut offset = [0u8; 8];
        reader.read_exact(&mut offset)?;

        match atoi::<u64>(&offset.trim_ascii_start()) {
            Some(offset) => header_offsets[i] = offset,
            None => {
                return Err(FCSError::InvalidHeader);
            }
        };
    }

    Ok(header_offsets)
}

/// Parse FCS metadata from text segment
fn parse_fcs_metadata(reader: &mut BufReader<File>, offsets: &[u64]) -> Result<HashMap<String, String>, FCSError> {

    let mut txt_segment = vec![0u8; (offsets[1] - offsets[0]) as usize];
    reader.seek(SeekFrom::Start(offsets[0]))?;
    reader.read_exact(&mut txt_segment)?;

    let txt_segment = String::from_utf8_lossy(&txt_segment);
    let delimiter = match txt_segment.chars().next() {
        Some(delimiter) => delimiter,
        None => return Err(FCSError::InvlidMetadata),
    };

    let mut value = String::new();
    let mut keyword = String::new();
    let mut metadata: HashMap<String, String> = HashMap::new();
    for s in txt_segment.split(delimiter) {

        if s.starts_with("$") {

            if keyword.len() > 0 && value.len() > 0 {
                metadata.insert(keyword.to_uppercase(), value.to_owned());
                // there's probably a better way to do this
                value.clear();
                keyword.clear()
            }

            keyword = s.to_string();

        } else {
            value.push_str(s);
        }
    }

    Ok(metadata)
}

