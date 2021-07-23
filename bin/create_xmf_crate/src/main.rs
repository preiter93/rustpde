#[macro_use]
extern crate failure;
pub mod sort_files;
pub mod utils;
pub mod xdmf_writer;
use failure::Error;
use sort_files::sorted_list_of_h5_files;
use std::path::Path;
use xdmf_writer::XdmfWriter;

/// Create xmf files from all files in root with attr and variable name
/// Example
/// ```ignore
/// create_xmf(
///    "data/",
///    &vec!["temp", "ux", "uy", "pres"],
///    &vec!["temp/v", "ux/v", "uy/v", "pres/v"],
/// )?;
/// ```
pub fn create_xmf<P>(root: P, attr: &[&str], var: &[&str]) -> Result<(), Error>
where
    P: AsRef<Path> + std::fmt::Display,
{
    let vec = sorted_list_of_h5_files(&root)?;
    for (i, v) in vec.iter().enumerate() {
        let xmfname = format!("{:}xmf{:0>6}.xmf", root, i);
        let fname = v.1.to_str().unwrap();
        let xdmf_writer = XdmfWriter::default(fname, attr, var, Some(&xmfname));
        xdmf_writer.create_cartesian(false)?;
        xdmf_writer.write()?;
        println!("Created xmf for {:?} => {:?}", fname, xmfname);
    }

    Ok(())
}

fn main() -> Result<(), Error> {
    let mut input = String::new();
    println!("Specify Directory: ");
    std::io::stdin()
        .read_line(&mut input)
        .expect("error: unable to read user input");

    let mut input = String::from(input.trim());
    if !input.ends_with("/") {
        input += "/";
    }

    create_xmf(
        &input,
        &["temp", "ux", "uy", "pres"],
        &["temp/v", "ux/v", "uy/v", "pres/v"],
    )?;

    Ok(())
}
