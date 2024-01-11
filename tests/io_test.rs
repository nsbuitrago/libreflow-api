use libreflow_api::io::{read_fcs, Sample, FCSError};
use std::fs::read_to_string;

const FORMAT_3_1_TESTFILE: &str = "./tests/test_fcs_3_1.fcs";
const FORMAT_3_1_METADATA: &str = "./tests/test_fcs_3_1_metadata.csv";
const CSV_DELIMITER: &str = ";";

#[test]
pub fn test_fcs_reader() -> Result<(), FCSError> {

    let sample = read_fcs(FORMAT_3_1_TESTFILE)?;

    for line in read_to_string(FORMAT_3_1_METADATA)?.lines() {
        let mut split = line.split(CSV_DELIMITER);
        let key = split.next().unwrap();
        let value = split.next().unwrap();
        assert_eq!(sample.metadata.get(key).unwrap(), value);
    }

    Ok(())
}



