use std::num::ParseIntError;

use derive_more::{Display, From};

/// FCS Result.
pub type Result<T> = core::result::Result<T, Error>;

/// FCS Errors
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
