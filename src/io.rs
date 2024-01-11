use std::path::Path;
use std::fs::File;
use std::io::{self, BufReader, Read, Seek, SeekFrom};
use byteorder::{ByteOrder, LittleEndian, BigEndian};
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
    #[error("Invalid FCS metadata. Missing required keyword: `{0}`")]
    MissingRequiredKeyword(String),
    #[error("Invalid FCS data mode: `{0}`. Only list mode data is supported.")]
    InvalidDataMode(String),
    #[error("Parse Int error: {0}")]
    ParseIntError(#[from] std::num::ParseIntError),
    #[error("Invalid FCS metadata. Unrecognized byte order: `{0}`")]
    UnrecognizedByteOrder(String),
    #[error("Invalid FCS data: `{0}`")]
    InvalidData(String),
}

/// Required non-parameter indexed keywords for fcs text segment. Parameter keywords are checked
/// later using the $PAR keyword from here.
/// before parsing event data.
const REQUIRED_KEYWORDS: [&str; 12] = [
    "$BEGINANALYSIS", // byte-offset to the beginning of analysis segment
    "$BEGINDATA", // byte-offset of beginning of data segment
    "$BEGINSTEXT", // byte-offset to beginning of text segment
    "$BYTEORD", // byte order for data acquisition computer
    "$DATATYPE", // type of data in data segment (ASCII, int, float)
    "$ENDANALYSIS", // byte-offset to end of analysis segment
    "$ENDDATA", // byte-offset to end of data segment
    "$ENDSTEXT", // byte-offset to end of text segment
    "$MODE", // data mode (list mode - preferred, histogram - deprecated)
    "$NEXTDATA", // byte-offset to next data set in the file
    "$PAR", // number of parameters in an event
    "$TOT" // total number of events in the data set
];

/// FCS sample holding metadata and event data
pub struct Sample {
    pub metadata: HashMap<String, String>,
    pub event_data: HashMap<String, Vec<f64>>,
}

/// Experiment holding 1 or more samples
pub struct Experiment {
    pub samples: Vec<Sample>,
}

impl Experiment {

    /// Load experiment data from directory
    pub fn load<P: AsRef<Path>>(exp_dir: P) -> Self {
        // get list of fcs files in directory
        // for each file, read metadata and event data into sample
        // add sample to experiment
        // return experiment
        //
        // let samples: Vec<Sample> = Vec::with_capacity(sample_paths.len());
    
        Self {
            samples: Vec::new(),
        }
    }

    pub fn save<P: AsRef<Path>>(&self, exp_dir: P) -> Result<(), FCSError> {
        // create directory if it doesn't exist
        // for each sample, create a directory
        // for each sample, write metadata to txt file
        // for each sample, write event data to csv file
        //
        // let sample_paths: Vec<PathBuf> = Vec::with_capacity(self.samples.len());
        //
        // for sample in self.samples {
        //     let sample_dir = exp_dir.join(sample.name);
        //     sample_paths.push(sample_dir);
        // }
        //
        // Ok(())
        Ok(())
    }
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
pub fn read_fcs<P: AsRef<Path>>(path: P) -> Result<Sample, FCSError> {
    let fcs_file = File::open(path)?;
    let mut reader = BufReader::new(fcs_file);
    let metadata = parse_fcs_metadata(&mut reader)?;
    let event_data = parse_fcs_event_data(&mut reader, &metadata)?;
    
    Ok(Sample {
        metadata: HashMap::new(),
        event_data: HashMap::new(),
    })
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
fn parse_fcs_metadata(reader: &mut BufReader<File>) -> Result<HashMap<String, String>, FCSError> {

    let offsets = parse_fcs_header(reader)?;
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

    check_required_fcs_keywords(&metadata)?;

    Ok(metadata)
}

/// Check that all required keywords are present in FCS metadata
fn check_required_fcs_keywords(metadata: &HashMap<String, String>) -> Result<(), FCSError> {

    for keyword in REQUIRED_KEYWORDS.iter() {
        if !metadata.contains_key(*keyword) {
            return Err(FCSError::MissingRequiredKeyword(keyword.to_string()));
        }
    }

    Ok(())
}

/// Parse FCS events from data segment
fn parse_fcs_event_data(reader: &mut BufReader<File>, metadata: &HashMap<String, String>) -> Result<HashMap<String, Vec<f64>>, FCSError> {
    
    let mode: &str = metadata.get("$MODE").unwrap(); // previously checked that this exists so we just unwrap here
    if mode != "L" {
        return Err(FCSError::InvalidDataMode(mode.to_string()));
    }

    let data_type: &str = metadata.get("$DATATYPE").unwrap();
    let n_params: usize = metadata.get("$PAR").unwrap().parse()?;
    let n_events: usize = metadata.get("$TOT").unwrap().parse()?;
    let byte_order: &str = metadata.get("$BYTEORD").unwrap();
    let capacity: usize = n_events * n_params;

    let mut event_data: HashMap<String, Vec<f64>> = HashMap::with_capacity(n_params);
    let mut events: Vec<f64> = Vec::with_capacity(capacity);

    if capacity == 0 {
        return Ok(event_data)
    }

    let data_segment_start: u64 = metadata.get("$BEGINDATA").unwrap().parse()?;
    let data_segment_end: u64 = metadata.get("$ENDDATA").unwrap().parse()?;
    let data_segment_size: u64 = data_segment_end - data_segment_start;
    reader.seek(SeekFrom::Start(data_segment_start))?;

    // FIXME: byte order will be the same for all parameters I think, so we should refactor this to check
    // byte ordre only once per parameter
    for i in 1..=n_params {
        match byte_order {
            "1,2,3,4" => {
                events = get_events::<LittleEndian>(reader, data_type, n_events, metadata, i)?;
            },
            "4,3,2,1" => {
                events = get_events::<BigEndian>(reader, data_type, n_events, metadata, i)?;
            },
            _ => {
                return Err(FCSError::UnrecognizedByteOrder(byte_order.to_string()));
            }
        }

        let id = metadata.get(&format!("$P{}S", i)).unwrap().to_string();
        event_data.insert(id, events);
    }

    Ok(event_data)
}

fn get_events<B: byteorder::ByteOrder>(reader: &mut BufReader<File>, data_type: &str, n_events: usize, metadata: &HashMap<String, String>, param_idx: usize) -> Result<Vec<f64>, FCSError> {
    // FIXME: this is a mess. We need to refactor this to be more readable and less repetitive.
    let data = match data_type {
        "F" => {
            let mut data = Vec::with_capacity(n_events);
            let mut float_buffer = vec![0; n_events * std::mem::size_of::<f32>()];
            reader.read_exact(&mut float_buffer)?;
            for i in 0..n_events {
                let float_value = B::read_f32(&float_buffer[i * 4..(i + 1) * 4]) as f64;
                data.push(float_value);
            }
            data
        },
        "D" => {
            let mut data = Vec::with_capacity(n_events);
            for i in 0..n_events {
                let float_buffer = vec![0; n_events * std::mem::size_of::<f64>()];
                let double = B::read_f64(&float_buffer[i * 8..(i+1) * 8]);
                data.push(double);
            }
            data
        },
        "I" => {
            // just unwrap the things since we checked they are present already.
            let bits_per_param = metadata.get(&format!("$P{}B", param_idx)).unwrap().parse::<usize>().unwrap();
            match bits_per_param / 8 {
                2 => {
                    let mut data = Vec::with_capacity(n_events);
                    for i in 0..n_events {
                        let events_u16 = vec![0; n_events * std::mem::size_of::<u16>()];
                        let double = B::read_u16(&events_u16[i * 2..(i+1) * 2]) as f64;
                        data.push(double);
                    }
                    data
                },
                4 => {
                    let mut data = Vec::with_capacity(n_events);
                    for i in 0..n_events {
                        let events_u32= vec![0; n_events * std::mem::size_of::<u32>()];
                        let double = B::read_u16(&events_u32[i * 4..(i+1) * 4]) as f64;
                        data.push(double);
                    }
                    data
                },
                8 => {
                    let mut data = Vec::with_capacity(n_events);
                    for i in 0..n_events {
                        let events_u64= vec![0; n_events * std::mem::size_of::<u64>()];
                        let double = B::read_u16(&events_u64[i * 8..(i+1) * 8]) as f64;
                        data.push(double);
                    }
                    data
                },
                16 => {
                    let mut data = Vec::with_capacity(n_events);
                    for i in 0..n_events {
                        let events_u128= vec![0; n_events * std::mem::size_of::<u128>()];
                        let double = B::read_u16(&events_u128[i * 16..(i+1) * 16]) as f64;
                        data.push(double);
                    }
                    data
                },
                _ => return Err(FCSError::InvalidData(("Bits for param type not supported").to_string())),
            }
        }
        _ => return Err(FCSError::InvalidData("FCS data type not supported. Must be F, D, or I".to_string()))
    };

    Ok(data)
}

