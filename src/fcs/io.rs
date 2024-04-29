use crate::fcs::{EventData, Metadata, Sample};
use atoi::atoi;
use byteorder::ReadBytesExt;
use derive_more::{Display, From};
use nom::bytes::complete::{is_not, tag, take};
use nom::combinator::map_res;
use nom::error::ErrorKind;
use nom::multi::fold_many1;
use nom::sequence::{separated_pair, terminated, tuple};
use nom::IResult;
use regex::Regex;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::num::ParseIntError;
use std::ops::RangeInclusive;
use std::path::Path;
use std::str::FromStr;

/// FCS IO Error
#[derive(Display, From, Debug)]
pub enum Error {
    #[from]
    IO(std::io::Error),

    #[display("Invalid FCS version: {}", version)]
    InvalidVersion {
        version: String,
    },

    #[display("Invalid file type found. File must be fcs.")]
    InvalidFileType,

    #[display("Failed to parse header segment offset.")]
    FailedHeaderOffsetParse,

    #[display("Failed to parse text segment delimiter.")]
    FailedDelimiterParse,

    #[display("Metadata and header segment offsets don't match.")]
    MetadataOffsetMismatch,

    FailedMetadataParse,

    #[from]
    FailedIntParse(ParseIntError),

    InvalidMetadata,

    #[display("Invalid data mode: {data_mode} for version {version}")]
    InvalidDataMode {
        data_mode: String,
        version: String,
    },

    #[display("Invalid data type: {kind} for version {version}")]
    InvalidDataType {
        kind: String,
        version: String,
    },

    #[display("Could not find key: {key}, in FCS metadata")]
    MetadataKeyNotFound {
        key: String,
    },

    NoDataFound,

    #[display("Invalid bit param length: {bit_length} for parameter index {index}")]
    InvalidParamBitLength {
        bit_length: usize,
        index: usize,
    },

    InvalidByteOrder {
        byte_order: String,
    },

    #[from]
    FromUtf8Error(std::string::FromUtf8Error),
}

/// FCS IO Result.
type Result<T> = core::result::Result<T, Error>;

/// Attempts to read FCS file and return Sample data
pub fn read<P: AsRef<Path>>(path: P) -> Result<Sample> {
    if path.as_ref().extension() != Some("fcs".as_ref()) {
        return Err(Error::InvalidFileType);
    }

    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    let header = read_header(&mut reader)?;
    let metadata = read_metadata(&mut reader, &header)?;
    let event_data = read_event_data(&mut reader, &metadata)?;

    Ok(Sample {
        metadata,
        event_data,
    })
}

/// Valid FCS versions
#[derive(Debug, PartialEq)]
enum Version {
    FCS3_1,
    FCS3_0,
}

impl FromStr for Version {
    type Err = Error;

    // Get version enum from string
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "FCS3.1" => Ok(Version::FCS3_1),
            // for now, we only support 3.1, but we leave this as a placeholder
            "FCS3.0" => Ok(Version::FCS3_0),
            _ => Err(Error::InvalidVersion {
                version: s.to_string(),
            }),
        }
    }
}

impl Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let str = match self {
            Version::FCS3_1 => "FCS3.1".to_string(),
            Version::FCS3_0 => "FCS3.0".to_string(),
        };
        write!(f, "{}", str)
    }
}

/// FCS header segment information.
struct Header {
    version: Version,
    text_offsets: RangeInclusive<usize>,
    data_offsets: RangeInclusive<usize>,
    analysis_offsets: RangeInclusive<usize>,
}

/// Read FCS header segment.
fn read_header(reader: &mut BufReader<File>) -> Result<Header> {
    let mut version_buffer = [0u8; 6];
    reader.read_exact(&mut version_buffer)?;
    let version = String::from_utf8(version_buffer.to_vec())?.parse::<Version>()?;

    reader.seek(SeekFrom::Current(4))?; // skip 4 bytes encoding whitespace

    let mut offset_buffer = [0u8; 48]; // 6 x 8 byte offsets
    reader.read_exact(&mut offset_buffer)?;

    let (offset_buffer, text_offsets) = parse_segment_offsets(&offset_buffer)?;
    let (offset_buffer, data_offsets) = parse_segment_offsets(&offset_buffer)?;
    let (_, analysis_offsets) = parse_segment_offsets(&offset_buffer)?;

    Ok(Header {
        version,
        text_offsets,
        data_offsets,
        analysis_offsets,
    })
}

/// Helper for parsing a single segment offset in header
fn parse_segment_offsets(input: &[u8]) -> Result<(&[u8], RangeInclusive<usize>)> {
    let (input, (start, stop)) = tuple((parse_offset_bytes, parse_offset_bytes))(input)
        .map_err(|_| Error::FailedHeaderOffsetParse)?;

    Ok((input, start..=stop))
}

/// Helper for parsing ascii encoded offset into an usize
fn parse_offset_bytes(input: &[u8]) -> IResult<&[u8], usize> {
    map_res(take(8usize), |bytes: &[u8]| {
        atoi::<usize>(bytes.trim_ascii_start()).ok_or(ErrorKind::Fail)
    })(input)
}

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
    "$TR",            // trigger parameter and its threshold
    "$VOL",           // volume of sample run during data acquisition
    "$WELLID",        // well identifier
];

/// Read FCS text segment.
fn read_metadata(reader: &mut BufReader<File>, header: &Header) -> Result<Metadata> {
    reader.seek(SeekFrom::Start(*header.text_offsets.start() as u64))?;
    let mut metadata_buf = vec![0u8; *header.text_offsets.end() - *header.text_offsets.start()];
    reader.read_exact(&mut metadata_buf)?;

    let metadata_txt = String::from_utf8(metadata_buf)?;

    let (metadata_txt, delimiter) =
        parse_delimiter(&metadata_txt).map_err(|_| Error::FailedDelimiterParse)?;

    // We handle double delimiters by replacing them with a temporary string.
    // This is done simply because it's a pain to handle double delimiters
    // when each key/value is separated by a single delimiter.
    // We'll replace the temporary string with the delimiter after parsing.
    let metadata_txt = metadata_txt.replace(&delimiter.repeat(2), DOUBLE_DELIMITER_TRANSFORM);

    let (_, metadata) = fold_many1(
        |input| parse_metadata_pairs(input, delimiter),
        HashMap::new,
        |mut acc: HashMap<String, String>, (key, value)| {
            acc.insert(key, value);
            acc
        },
    )(&metadata_txt)
    .map_err(|_| Error::FailedMetadataParse)?;

    metadata.is_valid()?;
    cross_validate(&metadata, &header)?;
    Ok(metadata)
}

/// Parse text segment delimiter
fn parse_delimiter(input: &str) -> IResult<&str, &str> {
    take(1u8)(input)
}

/// Metadata string parser
fn parse_metadata_string<'a>(input: &'a str, delimiter: &str) -> IResult<&'a str, String> {
    map_res(is_not(delimiter), |s: &str| {
        // Here, we replace the temporary string with the delimiter after extracting
        // the key or value string.
        Ok::<String, std::io::Error>(s.replace(DOUBLE_DELIMITER_TRANSFORM, delimiter))
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

/// Check recovered segment offsets from metadata match those in header segment
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

trait GetRequiredKey {
    fn get_required_key(&self, key: &str) -> Result<&str>;
}

impl GetRequiredKey for Metadata {
    /// Attempt to get a required key from the metadata hashmap, but return an
    /// FCS IO Result rather than option better error handling.
    fn get_required_key(&self, key: &str) -> Result<&str> {
        self.get(key)
            .ok_or(Error::MetadataKeyNotFound {
                key: key.to_string(),
            })
            .map(|s| s.as_str())
    }
}

/// Assert that types that implement this trait are valid
trait IsValid {
    fn is_valid(&self) -> Result<()>;
}

impl IsValid for Metadata {
    /// Assert that recovered metadata from the FCS text segment is valid.
    fn is_valid(&self) -> Result<()> {
        // this is a required key, so we just return an error if not found
        let n_params = self.get_required_key("$PAR")?;

        let n_digits = n_params.chars().count().to_string();
        let parameter_indexed_regex = r"[PR]\d{1,".to_string() + &n_digits + "}[BENRDFGLOPSTVIW]";

        // this is safe to unwrap since regex has to be valid
        let param_keywords = Regex::new(&parameter_indexed_regex).unwrap();

        // check that keys are valid
        for key in self.keys() {
            if !REQUIRED_KEYWORDS.contains(&key.as_str())
                && !param_keywords.is_match(key)
                && !OPTIONAL_KEYWORDS.contains(&key.as_str())
            {
                return Err(Error::InvalidMetadata);
            }
        }

        Ok(())
    }
}

/// Assert recovered metadata is consistent with header information
fn cross_validate(metadata: &Metadata, header: &Header) -> Result<()> {
    // check that data segment offsets from header match those in metadata
    let begin_data = metadata.get_required_key("$BEGINDATA")?;
    let end_data = metadata.get_required_key("$ENDDATA")?;
    validate_metadata_offsets(
        begin_data.parse::<usize>()?,
        end_data.parse::<usize>()?,
        &header.data_offsets,
    )?;

    // check that analysis segment offsets from header match those in metadata
    let begin_analysis = metadata.get_required_key("$BEGINANALYSIS")?;
    let end_analysis = metadata.get_required_key("$ENDANALYSIS")?;
    validate_metadata_offsets(
        begin_analysis.parse::<usize>()?,
        end_analysis.parse::<usize>()?,
        &header.analysis_offsets,
    )?;

    // validate some version specific metadata
    match header.version {
        Version::FCS3_1 => {
            let data_mode = metadata.get_required_key("$MODE")?;
            if data_mode != "L" {
                return Err(Error::InvalidDataMode {
                    data_mode: data_mode.to_string(),
                    version: header.version.to_string(),
                });
            }

            let data_type = metadata.get_required_key("$DATATYPE")?;
            if data_type != "I" && data_type != "F" && data_type != "D" {
                return Err(Error::InvalidDataType {
                    kind: data_type.to_string(),
                    version: header.version.to_string(),
                });
            }
        }
        Version::FCS3_0 => {
            todo!()
        }
    }
    Ok(())
}

/// Parse FCS data segment.
fn read_event_data(
    reader: &mut BufReader<std::fs::File>,
    metadata: &Metadata,
) -> Result<EventData> {
    let n_params = metadata.get_required_key("$PAR")?.parse::<usize>()?;
    let n_events = metadata.get_required_key("$TOT")?.parse::<usize>()?;
    let capacity = n_params * n_events;

    if capacity == 0 {
        return Err(Error::NoDataFound);
    }

    let byte_order = metadata.get_required_key("$BYTEORD")?;
    let data_type = metadata.get_required_key("$DATATYPE")?;
    let data_start = metadata.get_required_key("$BEGINDATA")?.parse::<u64>()?;

    reader.seek(SeekFrom::Start(data_start))?;
    let mut events: Vec<f64>;
    let mut data: HashMap<String, Vec<f64>> = HashMap::with_capacity(n_params);

    match metadata.get_required_key("$MODE")? {
        // List mode
        "L" => {
            for i in 1..=n_params {
                match byte_order {
                    "1,2,3,4" => {
                        events = parse_events::<byteorder::LittleEndian>(
                            reader, &data_type, n_events, metadata, i,
                        )?;
                    }
                    "4,3,2,1" => {
                        events = parse_events::<byteorder::BigEndian>(
                            reader, &data_type, n_events, metadata, i,
                        )?;
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
        "H" => todo!(),
        _ => unreachable!(),
    }
}

fn parse_events<B: byteorder::ByteOrder>(
    reader: &mut BufReader<std::fs::File>,
    data_type: &str,
    n_events: usize,
    metadata: &Metadata,
    index: usize,
) -> Result<Vec<f64>> {
    let mut data: Vec<f64> = Vec::with_capacity(n_events);
    match data_type {
        // unsigned binary integer type
        "I" => {
            let bit_length = metadata
                .get_required_key(&format!("P{}B", index))?
                .parse::<usize>()?;
            match bit_length {
                16 => {
                    for _ in 0..n_events {
                        let event = reader.read_u16::<B>()? as f64;
                        data.push(event);
                    }
                }
                32 => {
                    for _ in 0..n_events {
                        let event = reader.read_u32::<B>()? as f64;
                        data.push(event);
                    }
                }
                64 => {
                    for _ in 0..n_events {
                        let event = reader.read_u64::<B>()? as f64;
                        data.push(event);
                    }
                }
                128 => {
                    for _ in 0..n_events {
                        let event = reader.read_u128::<B>()? as f64;
                        data.push(event);
                    }
                }
                _ => return Err(Error::InvalidParamBitLength { bit_length, index }),
            }
        }
        // single precision floating point
        "F" => {
            for _ in 0..n_events {
                let event = reader.read_f32::<B>()? as f64;
                data.push(event);
            }
        }
        // double precision floating point
        "D" => {
            for _ in 0..n_events {
                let event = reader.read_f64::<B>()?;
                data.push(event);
            }
        }
        "A" => {
            unimplemented!()
        }
        _ => unreachable!(),
    }
    Ok(data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fcs_header_parser() -> Result<()> {
        let file = File::open("tests/data/test_fcs_3_1.fcs")?;
        let mut reader = BufReader::new(file);

        let header = read_header(&mut reader)?;

        assert_eq!(header.version, Version::FCS3_1);
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
        let sample = read("tests/data/test_fcs_3_1.fcs")?;

        let n_params = sample.metadata.get_required_key("$PAR")?.parse::<usize>()?;
        let n_param_vecs = sample.event_data.len();
        assert_eq!(n_params, n_param_vecs);

        let n_events = sample.metadata.get_required_key("$TOT")?.parse::<usize>()?;
        let param_id = sample.metadata.get_required_key("$P1N")?;
        let param_data = sample.event_data.get(param_id).unwrap();
        assert_eq!(n_events, param_data.len());

        Ok(())
    }
}
