use crate::utils::hdf5_get_scalar;
use failure::Error;
use std::path::Path;
use std::path::PathBuf;

/// Return list of files in root with specific ending
pub fn list_of_files_of_type<P: AsRef<Path>>(root: P, ending: &str) -> Result<Vec<PathBuf>, Error> {
    Ok(std::fs::read_dir(root)?
        .into_iter()
        .filter(|r| r.is_ok())
        .map(|r| r.unwrap().path())
        .filter(|r| r.extension() == Some(std::ffi::OsStr::new(ending)))
        .collect())
}

/// Retrieve time information from list of h5 files
pub fn get_time_of_h5_files(list: &[PathBuf]) -> Result<Vec<f64>, Error> {
    Ok(list
        .iter()
        .map(|f| hdf5_get_scalar(f, "time").unwrap())
        .collect())
}

/// Get sorted list of h5 files (sorted by time)
pub fn sorted_list_of_h5_files<P: AsRef<Path>>(root: P) -> Result<Vec<(f64, PathBuf)>, Error> {
    let list = list_of_files_of_type(root, "h5")?;
    let time = get_time_of_h5_files(&list)?;
    let mut vec = time.iter().cloned().zip(list).collect::<Vec<_>>();
    vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    Ok(vec)
}
