use crate::error::IoError;
use ndarray::{Array2, Array3, Array4, Array5};

pub(super) fn vec2_to_array2(vec: &[Vec<f32>]) -> Result<Array2<f32>, IoError> {
    let rows = vec.len();
    let cols = if rows > 0 { vec[0].len() } else { 0 };
    let flat: Vec<f32> = vec.iter().flat_map(|row| row.iter().cloned()).collect();
    Array2::from_shape_vec((rows, cols), flat).map_err(|e| {
        IoError::StdIoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })
}

pub(super) fn vec3_to_array3(vec: &[Vec<Vec<f32>>]) -> Result<Array3<f32>, IoError> {
    let d0 = vec.len();
    let d1 = if d0 > 0 { vec[0].len() } else { 0 };
    let d2 = if d1 > 0 && d0 > 0 { vec[0][0].len() } else { 0 };
    let flat: Vec<f32> = vec
        .iter()
        .flat_map(|v1| v1.iter().flat_map(|v2| v2.iter().cloned()))
        .collect();
    Array3::from_shape_vec((d0, d1, d2), flat).map_err(|e| {
        IoError::StdIoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })
}

pub(super) fn vec4_to_array4(vec: &[Vec<Vec<Vec<f32>>>]) -> Result<Array4<f32>, IoError> {
    let d0 = vec.len();
    let d1 = if d0 > 0 { vec[0].len() } else { 0 };
    let d2 = if d1 > 0 && d0 > 0 { vec[0][0].len() } else { 0 };
    let d3 = if d2 > 0 && d1 > 0 && d0 > 0 {
        vec[0][0][0].len()
    } else {
        0
    };
    let flat: Vec<f32> = vec
        .iter()
        .flat_map(|v1| {
            v1.iter()
                .flat_map(|v2| v2.iter().flat_map(|v3| v3.iter().cloned()))
        })
        .collect();
    Array4::from_shape_vec((d0, d1, d2, d3), flat).map_err(|e| {
        IoError::StdIoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })
}

pub(super) fn vec5_to_array5(vec: &[Vec<Vec<Vec<Vec<f32>>>>]) -> Result<Array5<f32>, IoError> {
    let d0 = vec.len();
    let d1 = if d0 > 0 { vec[0].len() } else { 0 };
    let d2 = if d1 > 0 && d0 > 0 { vec[0][0].len() } else { 0 };
    let d3 = if d2 > 0 && d1 > 0 && d0 > 0 {
        vec[0][0][0].len()
    } else {
        0
    };
    let d4 = if d3 > 0 && d2 > 0 && d1 > 0 && d0 > 0 {
        vec[0][0][0][0].len()
    } else {
        0
    };
    let flat: Vec<f32> = vec
        .iter()
        .flat_map(|v1| {
            v1.iter().flat_map(|v2| {
                v2.iter()
                    .flat_map(|v3| v3.iter().flat_map(|v4| v4.iter().cloned()))
            })
        })
        .collect();
    Array5::from_shape_vec((d0, d1, d2, d3, d4), flat).map_err(|e| {
        IoError::StdIoError(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            e.to_string(),
        ))
    })
}
