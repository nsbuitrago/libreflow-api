use libreflow_api::io::{read_fcs, FCSError}; 


const FORMAT_3_1_TESTFILE: &str = "./tests/test_file.fcs";

#[test]
pub fn test_fcs_reader() -> Result<(), FCSError> {
    read_fcs(FORMAT_3_1_TESTFILE)?;
    Ok(())
}
