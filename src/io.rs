use std::path::Path;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use thiserror::Error;
use atoi::atoi;

const VALID_FCS_VERSIONS: [&[u8; 6]; 2] = [b"FCS3.0", b"FCS3.1"];

const N_FCS_OFFSETS: usize = 2; // we set this to 2 since we only care about the txt segment

#[derive(Debug, Error)]
pub enum FCSError {
    #[error("IO Error: {0}")]
    IOError(#[from] io::Error)
}

/// Read FCS file data and return as Sample
pub fn read_fcs<P: AsRef<Path>>(path: P) -> Result<(), FCSError> {
    // open fcs file
    let fcs_file = File::open(path)?;
    let mut reader = BufReader::new(fcs_file);

    // parse header
    let _header = parse_header(&mut reader)?;

    Ok(())
}

/// Parse FCS header and return offsets to txt, data, and analysis segments
pub fn parse_header(reader: &mut BufReader<File>) -> Result<[u64; N_FCS_OFFSETS], FCSError> {
    let mut fcs_version = [0u8; 6];
    reader.read_exact(&mut fcs_version)?;

    if !VALID_FCS_VERSIONS.contains(&&fcs_version) {
        println!("Invalid FCS version");
    } 

    reader.seek(SeekFrom::Current(4))?;
    let mut header_offsets = [0u64; N_FCS_OFFSETS];

    for i in 0..N_FCS_OFFSETS {
        let mut offset = [0u8; 8];
        reader.read_exact(&mut offset)?;

        match atoi::<u64>(&offset.trim_ascii_start()) {
            Some(offset) => header_offsets[i] = offset,
            None => {
                println!("Invalid offset");
                continue;
            }
        };
    }
    println!("Header offsets: {:?}", header_offsets);

    Ok(header_offsets)
}
