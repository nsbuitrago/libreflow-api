#![allow(dead_code, unused)]
use atoi::atoi;
use derive_more::{Display, From};
use nom::{
    branch::alt,
    bytes::complete::{escaped_transform, is_not, tag, take, take_while},
    combinator::{map_res, value},
    error::ErrorKind,
    multi::fold_many1,
    sequence::{pair, preceded, separated_pair, terminated, tuple},
    IResult,
};
use std::{
    collections::HashMap,
    io::{self, BufReader, Read, Seek, SeekFrom},
    ops::RangeInclusive,
    str::Utf8Error,
};

use crate::fcs::File;

/// Currently supported FCS versions.
const VALID_FCS_VERSIONS: [&[u8; 6]; 2] = [b"FCS3.0", b"FCS3.1"];

/// Escaped delimiters in keys or values in the text segment are replaced with
/// this temporary string during parsing. This is done to simplify parsing.
/// The temporary string is replaced with a single delimiter after parsing.
const DOUBLE_DELIMITER_TRANSFORM: &str = "@ESCAPED@";

/// FCS IO errors
#[derive(Display, From, Debug)]
pub enum Error {
    InvalidFileType,
    Io(std::io::Error),
    InvalidFCSVersion,
    FcsParseError(Utf8Error),
}

/// FCS IO result
pub type Result<T> = std::result::Result<T, Error>;

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
        metadata.insert(String::from("version"), header.version);
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
            String::from_utf8(version.to_vec()).unwrap()
        } else {
            return Err(Error::InvalidFCSVersion);
        };

        reader.seek(SeekFrom::Current(4))?; // skip 4 bytes encoding whitespace
        let mut offset_bytes = [0u8; 48];
        reader.read_exact(&mut offset_bytes)?;

        // FIXME: return nom error wrapped in Error
        let (offset_bytes, text_offsets) = parse_segment(&offset_bytes).unwrap();
        let (offset_bytes, data_offsets) = parse_segment(&offset_bytes).unwrap();
        let (_, analysis_offsets) = parse_segment(&offset_bytes).unwrap();

        Ok(Header {
            version,
            text_offsets,
            data_offsets,
            analysis_offsets,
        })
    }
}

/// Helper for parsing a single segment offset
fn parse_segment(input: &[u8]) -> IResult<&[u8], RangeInclusive<usize>> {
    let (input, (start, stop)) = tuple((parse_offset_bytes, parse_offset_bytes))(input)?;
    Ok((input, start..=stop))
}

/// Helper for parsing ascii encoded offset into a usize
fn parse_offset_bytes(input: &[u8]) -> IResult<&[u8], usize> {
    map_res(take(8usize), |bytes: &[u8]| {
        atoi::<usize>(bytes.trim_ascii_start()).ok_or(ErrorKind::Fail)
    })(input)
}

/// FCS metadata object
pub struct Metadata {
    text_offsets: RangeInclusive<usize>,
    data_offsets: RangeInclusive<usize>,
    analysis_offsets: RangeInclusive<usize>,
    pub text_segment: HashMap<String, String>,
}

impl Metadata {
    /// Parse FCS metadata given a bufreader and header information.
    pub fn parse(
        reader: &mut BufReader<&std::fs::File>,
        header: &Header,
    ) -> Result<HashMap<String, String>> {
        reader.seek(SeekFrom::Start(*header.text_offsets.start() as u64))?;
        let n_metadata_bytes = (*header.text_offsets.end() - *header.text_offsets.start()) as usize;
        let mut metadata_bytes = vec![0u8; n_metadata_bytes];
        reader.read_exact(&mut metadata_bytes)?;

        // FIXME unwrap here
        let metadata_text = String::from_utf8(metadata_bytes).unwrap();

        // Parse metadata text
        let (metadata_text, delimiter) = parse_delimiter(&metadata_text).unwrap(); // FIXME unwrap

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
        .unwrap(); // FIXME unwrap

        Ok(metadata)
    }

    /// Check that all required keys are present
    fn is_valid(&self) -> Result<()> {
        Ok(())
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
}
