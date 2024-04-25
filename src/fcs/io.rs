#![allow(dead_code, unused)]
use atoi::atoi;
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

use crate::fcs::File;

/// Currently supported FCS versions.
const VALID_FCS_VERSIONS: [&[u8; 6]; 2] = [b"FCS3.0", b"FCS3.1"];

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
    FromUtf8Error(FromUtf8Error),
    InvalidMetadata,
    FailedMetadataParse,
    FailedHeaderParse(String),
    #[from]
    FailedNomParse(String),
    #[from]
    FaileIntParse(ParseIntError),
    FailedDelimiterParse,
}

/// FCS IO result
pub type Result<T> = core::result::Result<T, Error>;

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

    pub fn parse(&self) -> Result<()> {
        let mut reader = BufReader::new(&self.inner);
        let header = Header::parse(&mut reader)?;
        let mut metadata = Metadata::parse(&mut reader, &header)?;
        metadata.is_valid()?;
        metadata
            .text_segment
            .insert(String::from("version"), header.version);
        // let data = Data::parse(&mut reader, &header.data_offsets)?;

        Ok(())
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
        let (offset_bytes, data_offsets) = parse_segment(&offset_bytes)?;
        let (_, analysis_offsets) = parse_segment(&offset_bytes)?;

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
pub struct Metadata<'a> {
    text_offsets: &'a RangeInclusive<usize>,
    data_offsets: &'a RangeInclusive<usize>,
    analysis_offsets: &'a RangeInclusive<usize>,
    pub text_segment: HashMap<String, String>,
}

impl<'a> Metadata<'a> {
    /// Parse FCS metadata given a bufreader and header information.
    pub fn parse(
        reader: &mut BufReader<&std::fs::File>,
        header: &'a Header,
    ) -> Result<Metadata<'a>> {
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
            text_offsets: &header.text_offsets,
            data_offsets: &header.data_offsets,
            analysis_offsets: &header.analysis_offsets,
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
            .ok_or(Error::InvalidMetadata)
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
        let data = file.parse();
        println!("{:?}", data);

        Ok(())
    }
}
