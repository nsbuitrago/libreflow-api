#![allow(dead_code, unused)]
use atoi::atoi;
use byteorder::{ByteOrder, ReadBytesExt};
use derive_more::{Display, From};
use nom::{
    branch::alt,
    bytes::complete::{escaped_transform, is_not, tag, take, take_while},
    combinator::{map_res, value},
    error::{ErrorKind, ParseError},
    multi::fold_many1,
    sequence::{pair, preceded, separated_pair, terminated, tuple},
    IResult,
};
use regex::Regex;
use std::{
    collections::HashMap,
    io::{self, BufReader, Read, Seek, SeekFrom},
    num::ParseIntError,
    ops::RangeInclusive,
    str::Utf8Error,
    string::FromUtf8Error,
};

use crate::fcs::Sample;

/// Currently supported FCS versions.
const VALID_FCS_VERSIONS: [&[u8; 6]; 1] = [b"FCS3.1"];

/// Escaped delimiters in keys or values in the text segment are replaced with
/// this temporary string during parsing. This is done to simplify parsing.
/// The temporary string is replaced with a single delimiter after parsing.
const DOUBLE_DELIMITER_TRANSFORM: &str = "@ESCAPED@";

/// Required non-parameter indexed keywords in the text segment.
const REQUIRED_KEYWORDS: [&str; 12] = [
    "$BEGINANALYSIS", // byte-offset to the beginning of analysis segment
    "$BEGINDATA",     // byte-offset of beginning of data segment
    "$BEGINSTEXT",    // byte-offset to beginning of text segment
    "$BYTEORD",       // byte order for data acquisition computer
    "$DATATYPE",      // type of data in data segment (ASCII, int, float)
    "$ENDANALYSIS",   // byte-offset to end of analysis segment
    "$ENDDATA",       // byte-offset to end of data segment
    "$ENDSTEXT",      // byte-offset to end of text segment
    "$MODE",          // data mode (list mode - preferred, histogram - deprecated)
    "$NEXTDATA",      // byte-offset to next data set in the file
    "$PAR",           // number of parameters in an event
    "$TOT",           // total number of events in the data set
];

/// Optional non-paramater indexed keywords
const OPTIONAL_KEYWORDS: [&str; 31] = [
    "$ABRT",          // events lost due to acquisition electronic coincidence
    "$BTIM",          // clock time at beginning of data acquisition
    "$CELLS",         // description of objects measured
    "$COM",           // comment
    "$CSMODE",        // cell subset mode, number of subsets an object may belong
    "$CSVBITS",       // number of bits used to encode cell subset identifier
    "$CYT",           // cytometer type
    "$CYTSN",         // cytometer serial number
    "$DATE",          // date of data acquisition
    "$ETIM",          // clock time at end of data acquisition
    "$EXP",           // investigator name initiating experiment
    "$FIL",           // name of data file containing data set
    "$GATE",          // number of gating parameters
    "$GATING",        // region combinations used for gating
    "$INST",          // institution where data was acquired
    "$LAST_MODIFIED", // timestamp of last modification
    "$LAST_MODIFIER", // person performing last modification
    "$LOST",          // number events lost due to computer busy
    "$OP",            // name of flow cytometry operator
    "$ORIGINALITY",   // information whether FCS data set has been modified or not
    "$PLATEID",       // plate identifier
    "$PLATENAME",     // plate name
    "$PROJ",          // project name
    "$SMNO",          // specimen (i.e., tube) label
    "$SPILLOVER",     // spillover matrix
    "$SRC",           // source of specimen (cell type, name, etc.)
    "$SYS",           // type of computer and OS
    "$TIMESTEP",      // time step for time parameter
    "$TR",            // trigger paramter and its threshold
    "$VOL",           // volume of sample run during data acquisition
    "$WELLID",        // well identifier
];

/// FCS IO errors
#[derive(Display, From, Debug)]
pub enum Error {
    InvalidFileType,
    #[from]
    Io(std::io::Error),
    InvalidFCSVersion,
    #[from]
    FailedUtf8Parse(FromUtf8Error),
    InvalidMetadata,
    FailedMetadataParse,
    FailedHeaderParse(String),
    #[from]
    FailedNomParse(String),
    #[from]
    FaileIntParse(ParseIntError),
    FailedDelimiterParse,
    InvalidData(String),

    #[display("Metadata key not found: {}", key)]
    MetadataKeyNotFound {
        key: String,
    },

    #[display("Invalid data mode: {} for FCS version: {}", mode, version)]
    InvalidDataMode {
        mode: String,
        version: String,
    },

    #[display("Invalid data type: {} for FCS version: {}", data_type, version)]
    InvalidDataType {
        data_type: String,
        version: String,
    },

    #[display("Invalid byte order: {}", byte_order)]
    InvalidByteOrder {
        byte_order: String,
    },

    #[display("Invalid bit length {} for parameter index {}", bit_length, index)]
    InvalidParameterBitLength {
        bit_length: usize,
        index: usize,
    },
}

/// FCS IO result
pub type Result<T> = core::result::Result<T, Error>;

/// FCS file object
#[derive(Debug)]
pub struct File {
    inner: std::fs::File,
}

impl File {
    /// Attempts to open an FCS file in read-only mode.
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let fcs_extension = path.as_ref().extension();
        if fcs_extension != Some("fcs".as_ref()) {
            return Err(Error::InvalidFileType);
        }

        let inner = std::fs::File::open(path)?;

        Ok(Self { inner })
    }

    pub fn parse(&self) -> Result<Sample> {
        let mut reader = BufReader::new(&self.inner);
        let header = Header::parse(&mut reader)?;
        let mut metadata = Metadata::parse(&mut reader, &header)?;
        metadata.is_valid()?;
        let data = Data::parse(&mut reader, &header.data_offsets, &metadata)?;
        //let sample = Sample { metadata, data };

        Ok(Sample { metadata, data })
    }
}

pub struct Header {
    version: String,
    text_offsets: RangeInclusive<usize>,
    data_offsets: RangeInclusive<usize>,
    analysis_offsets: RangeInclusive<usize>,
}

impl Header {
    /// Parse FCS header
    fn parse(reader: &mut BufReader<&std::fs::File>) -> Result<Header> {
        let mut version = [0u8; 6];
        reader.read_exact(&mut version)?;

        let version = if VALID_FCS_VERSIONS.contains(&&version) {
            String::from_utf8(version.to_vec())?
        } else {
            return Err(Error::InvalidFCSVersion);
        };

        reader.seek(SeekFrom::Current(4))?; // skip 4 bytes encoding whitespace

        let mut offset_bytes = [0u8; 48];
        reader.read_exact(&mut offset_bytes)?;

        let (offset_bytes, text_offsets) = parse_segment(&offset_bytes)?;
        let (offset_bytes, data_offsets) = parse_segment(offset_bytes)?;
        let (_, analysis_offsets) = parse_segment(offset_bytes)?;

        Ok(Header {
            version,
            text_offsets,
            data_offsets,
            analysis_offsets,
        })
    }
}

/// Helper for parsing a single segment offset
fn parse_segment(input: &[u8]) -> Result<(&[u8], RangeInclusive<usize>)> {
    let (input, (start, stop)) = tuple((parse_offset_bytes, parse_offset_bytes))(input)
        .map_err(|_| Error::FailedHeaderParse("Could not parse header segment".to_string()))?;

    Ok((input, start..=stop))
}

/// Helper for parsing ascii encoded offset into a usize
fn parse_offset_bytes(input: &[u8]) -> IResult<&[u8], usize> {
    map_res(take(8usize), |bytes: &[u8]| {
        atoi::<usize>(bytes.trim_ascii_start()).ok_or(ErrorKind::Fail)
    })(input)
}

/// FCS metadata object
#[derive(Debug, Clone)]
pub struct Metadata {
    version: String,
    text_offsets: RangeInclusive<usize>,
    data_offsets: RangeInclusive<usize>,
    analysis_offsets: RangeInclusive<usize>,
    pub text_segment: HashMap<String, String>,
}

impl Metadata {
    /// Parse FCS metadata given a bufreader and header information.
    pub fn parse(reader: &mut BufReader<&std::fs::File>, header: &Header) -> Result<Metadata> {
        reader.seek(SeekFrom::Start(*header.text_offsets.start() as u64))?;
        let n_metadata_bytes = (*header.text_offsets.end() - *header.text_offsets.start());
        let mut metadata_bytes = vec![0u8; n_metadata_bytes];
        reader.read_exact(&mut metadata_bytes)?;

        // FIXME unwrap here
        let metadata_text = String::from_utf8(metadata_bytes)?;

        // Parse metadata text
        let (metadata_text, delimiter) =
            parse_delimiter(&metadata_text).map_err(|_| Error::FailedDelimiterParse)?;

        // We handle double delimiters by replacing them with a temporary string.
        // This is done simply because it's a pain to handle double delimiters
        // when each key/value is separated by a single delimiter.
        // We'll replace the temporary string with the delimiter after parsing.
        let metadata_text =
            metadata_text.replace(delimiter.repeat(2).as_str(), DOUBLE_DELIMITER_TRANSFORM);

        let (_, metadata) = fold_many1(
            |input| parse_metadata_pairs(input, delimiter),
            HashMap::new,
            |mut acc: HashMap<String, String>, (key, value)| {
                acc.insert(key, value);
                acc
            },
        )(&metadata_text)
        .map_err(|_| Error::FailedMetadataParse)?;

        Ok(Metadata {
            version: header.version.to_owned(),
            text_offsets: header.text_offsets.clone(),
            data_offsets: header.data_offsets.clone(),
            analysis_offsets: header.analysis_offsets.clone(),
            text_segment: metadata,
        })
    }

    /// Check that all recovered metadata keys are valid and segment offsets match.
    fn is_valid(&self) -> Result<()> {
        // this is a required key so we just return an error if not found
        let n_params = self
            .text_segment
            .get("$PAR")
            .ok_or(Error::InvalidMetadata)?;

        let n_digits = n_params.chars().count().to_string();
        let parameter_indexed_regex = r"[PR]\d{1,".to_string() + &n_digits + "}[BENRDFGLOPSTVIW]";
        // this is safe to unwrap since regex has to be valid
        let param_keywords = Regex::new(&parameter_indexed_regex).unwrap();

        // check that keys are valid
        for key in self.text_segment.keys() {
            if !REQUIRED_KEYWORDS.contains(&key.as_str())
                && !param_keywords.is_match(key)
                && !OPTIONAL_KEYWORDS.contains(&key.as_str())
            {
                return Err(Error::InvalidMetadata);
            }
        }

        // check that data segment offsets from header match those in metadata
        let begin_data = self.get_required_key("$BEGINDATA")?;
        let end_data = self.get_required_key("$ENDDATA")?;
        validate_metadata_offsets(
            begin_data.parse::<usize>()?,
            end_data.parse::<usize>()?,
            &self.data_offsets,
        )?;

        // check that analysis segment offsets from header match those in metadata
        let begin_analysis = self.get_required_key("$BEGINANALYSIS")?;
        let end_analysis = self.get_required_key("$ENDANALYSIS")?;
        validate_metadata_offsets(
            begin_analysis.parse::<usize>()?,
            end_analysis.parse::<usize>()?,
            &self.analysis_offsets,
        )?;

        Ok(())
    }

    fn get_required_key(&self, key: &str) -> Result<&str> {
        self.text_segment
            .get(key)
            .ok_or(Error::MetadataKeyNotFound {
                key: key.to_string(),
            })
            .map(|s| s.as_str())
    }
}

/// Metadata delimiter parser
fn parse_delimiter(input: &str) -> IResult<&str, &str> {
    take(1u8)(input)
}

/// Metadata string parser
fn parse_metadata_string<'a>(input: &'a str, delimiter: &str) -> IResult<&'a str, String> {
    map_res(is_not(delimiter), |s: &str| {
        // Here, we replace the temporary string with the delimiter after extracting
        // the key or value string.
        Ok::<String, io::Error>(s.replace(DOUBLE_DELIMITER_TRANSFORM, delimiter))
    })(input)
}

/// Metadata key-value pair parser
fn parse_metadata_pairs<'a>(input: &'a str, delimiter: &str) -> IResult<&'a str, (String, String)> {
    separated_pair(
        |input| parse_metadata_string(input, delimiter), // keys
        tag(delimiter),                                  // delimiter separating the pair
        terminated(
            // values (terminated by delimiter or end of string)
            |input| parse_metadata_string(input, delimiter),
            tag(delimiter),
        ),
    )(input)
}

fn validate_metadata_offsets(
    seg_start: usize,
    seg_end: usize,
    seg_offsets: &RangeInclusive<usize>,
) -> Result<()> {
    if seg_start != *seg_offsets.start() || seg_end != *seg_offsets.end() {
        return Err(Error::InvalidMetadata);
    }

    Ok(())
}

pub trait IsValid {
    /// Check if metadata is valid for the given FCS version.
    fn is_valid(&self, version: &str) -> Result<()>;
}

enum DataMode {
    List,
    CorrelatedHistogram,
    UncorrelatedHistogram,
}

enum DataType {
    UInt,
    Single, // single precision IEEE floating point
    Double, // double precision IEEE floating point
    Ascii,
}

/// FCS data object
struct Data {
    data_offsets: RangeInclusive<usize>,
}

impl Data {
    /// Parse data segment
    fn parse(
        reader: &mut BufReader<&std::fs::File>,
        data_offsets: &RangeInclusive<usize>,
        metadata: &Metadata,
    ) -> Result<HashMap<String, Vec<f64>>> {
        let version = &metadata.version;
        match version.as_str() {
            "FCS3.1" => {
                let data_mode = metadata.get_required_key("$MODE")?;
                let data_mode = match data_mode {
                    "L" => Ok(DataMode::List),
                    _ => Err(Error::InvalidDataMode {
                        mode: data_mode.to_string(),
                        version: version.to_owned(),
                    }),
                }?;

                let data_type = metadata.get_required_key("$DATATYPE")?;
                let data_type = match data_type {
                    "I" => Ok(DataType::UInt),
                    "F" => Ok(DataType::Single),
                    "D" => Ok(DataType::Double),
                    _ => Err(Error::InvalidDataType {
                        data_type: data_type.to_string(),
                        version: version.to_owned(),
                    }),
                }?;

                let n_params = metadata.get_required_key("$PAR")?.parse::<usize>()?;
                let n_events = metadata.get_required_key("$TOT")?.parse::<usize>()?;
                let capacity = n_params * n_events;

                if capacity == 0 {
                    return Err(Error::InvalidData("No data found".to_string()));
                }

                let byte_order = metadata.get_required_key("$BYTEORD")?;

                reader.seek(SeekFrom::Start(*metadata.data_offsets.start() as u64))?;
                let mut events = Vec::with_capacity(n_events);
                let mut data: HashMap<String, Vec<f64>> = HashMap::with_capacity(n_params);

                // If data type is I i need to check each paramater for max bit length
                // and int range used by the parameter
                // for F, need to assert that PnB keywords are set to 32
                // and PnE keywords are set to 0,0
                // for D, need to assert that PnB keywords are set to 64
                // and PnE keywords are set to 0,0
                for i in 1..=n_params {
                    match byte_order {
                        "1,2,3,4" => {
                            events = parse_event_data::<byteorder::LittleEndian>(
                                reader, &data_type, n_events, &metadata, i,
                            )?
                        }
                        "4,3,2,1" => {
                            events = parse_event_data::<byteorder::BigEndian>(
                                reader, &data_type, n_events, &metadata, i,
                            )?
                        }
                        _ => {
                            return Err(Error::InvalidByteOrder {
                                byte_order: byte_order.to_string(),
                            })
                        }
                    }

                    let id = metadata.get_required_key(&format!("$P{}N", i))?;
                    data.insert(id.to_string(), events);
                }
                Ok(data)
            }
            "FCS3.0" => {
                //let data_mode = metadata.get_required_key("$MODE")?;
                //let data_mode = match data_mode {
                //    "L" => Ok(DataMode::List),
                //    "H" => Ok(DataMode::CorrelatedHistogram),
                //    "U" => Ok(DataMode::UncorrelatedHistogram),
                //    _ => Err(Error::InvalidDataMode {
                //        mode: data_mode.to_string(),
                //        version: version.to_owned(),
                //    }),
                //}?;
                unimplemented!()
            }
            _ => unreachable!(), // we already checked that version is valid
        }
    }
}

fn parse_event_data<B: byteorder::ByteOrder>(
    reader: &mut BufReader<&std::fs::File>,
    data_type: &DataType,
    n_events: usize,
    metadata: &Metadata,
    index: usize,
) -> Result<Vec<f64>> {
    let mut data: Vec<f64> = Vec::with_capacity(n_events);
    match data_type {
        DataType::UInt => {
            let bit_length = metadata
                .get_required_key(&format!("P{}B", index))?
                .parse::<usize>()?;
            match bit_length {
                16 => {
                    for i in 0..n_events {
                        let event = reader.read_u16::<B>()? as f64;
                        data.push(event);
                    }
                }
                32 => {
                    for i in 0..n_events {
                        let event = reader.read_u32::<B>()? as f64;
                        data.push(event);
                    }
                }
                64 => {
                    for i in 0..n_events {
                        let event = reader.read_u64::<B>()? as f64;
                        data.push(event);
                    }
                }
                128 => {
                    for i in 0..n_events {
                        let event = reader.read_u128::<B>()? as f64;
                        data.push(event);
                    }
                }
                _ => return Err(Error::InvalidParameterBitLength { bit_length, index }),
            }
        }
        DataType::Single => {
            for i in 0..n_events {
                let event = reader.read_f32::<B>()? as f64;
                data.push(event);
            }
        }
        DataType::Double => {
            for i in 0..n_events {
                let event = reader.read_f64::<B>()?;
                data.push(event);
            }
        }
        DataType::Ascii => {
            unimplemented!()
        }
    }

    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fcs::io::{Error, File, Header, Result};

    #[test]
    fn open_fcs_file() {
        // wrong extension
        let file = File::open("tests/data/test_fcs_3_1.txt");
        assert!(file.is_err());

        // no extension
        let file = File::open("tests/data/test_fcs_3_1");
        assert!(file.is_err());

        // non-existent file
        let file = File::open("tests/data/non_existent.fcs");
        assert!(file.is_err());

        // valid file
        let file = File::open("tests/data/test_fcs_3_1.fcs");
        assert!(file.is_ok());
    }

    #[test]
    fn fcs_header_parser() -> Result<()> {
        let file = File::open("tests/data/test_fcs_3_1.fcs")?;
        let mut reader = BufReader::new(&file.inner);
        let header = Header::parse(&mut reader)?;

        assert_eq!(header.version, "FCS3.1");
        assert_eq!(header.text_offsets, 64..=1717);
        assert_eq!(header.data_offsets, 1718..=5201717);
        assert_eq!(header.analysis_offsets, 0..=0);

        Ok(())
    }

    #[test]
    fn fcs_metadata_parser() {
        let metadata_string =
            "\\Key1\\Value1\\Escaped\\\\Key2\\Value2\\Key3\\Escaped\\\\Value3\\Key 4\\Value-4\\";

        let true_metadata_map: HashMap<String, String> = HashMap::from_iter(vec![
            ("Key1".to_string(), "Value1".to_string()),
            ("Escaped\\Key2".to_string(), "Value2".to_string()),
            ("Key3".to_string(), "Escaped\\Value3".to_string()),
            ("Key 4".to_string(), "Value-4".to_string()),
        ]);

        let (metadata_string, delimiter) = parse_delimiter(&metadata_string).unwrap();
        let metadata_string_transformed =
            metadata_string.replace(delimiter.repeat(2).as_str(), DOUBLE_DELIMITER_TRANSFORM);

        let (_, metadata) = fold_many1(
            |input| parse_metadata_pairs(input, delimiter),
            HashMap::new,
            |mut acc: HashMap<String, String>, (key, value)| {
                acc.insert(key, value);
                acc
            },
        )(&metadata_string_transformed)
        .unwrap();

        assert_eq!(metadata, true_metadata_map);
    }

    #[test]
    fn full_fcs_parser() -> Result<()> {
        let file = File::open("tests/data/test_fcs_3_1.fcs")?;
        let sample = file.parse()?;

        let n_params = sample.metadata.get_required_key("$PAR")?.parse::<usize>()?;
        let n_param_vecs = sample.data.len();
        assert_eq!(n_params, n_param_vecs);

        let n_events = sample.metadata.get_required_key("$TOT")?.parse::<usize>()?;
        let param_id = sample.metadata.get_required_key("$P1N")?;
        let param_data = sample.data.get(param_id).unwrap();
        assert_eq!(n_events, param_data.len());

        Ok(())
    }
}
