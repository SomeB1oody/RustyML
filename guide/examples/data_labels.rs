use ndarray::array;
use rustyml::utils::{to_categorical, to_categorical_with_mapping, to_sparse_categorical};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let labels = array![0_i32, 2, 1];
    let encoded = to_categorical(&labels, Some(3))?;
    println!("one-hot shape: {:?}", encoded.dim());
    println!("first row: {:?}", encoded.row(0).to_vec());
    println!("decoded: {:?}", to_sparse_categorical(&encoded)?.to_vec());
    let (encoded, mapping) = to_categorical_with_mapping(&["cat", "dog", "cat"], None)?;
    println!("cat: {}, dog: {}", mapping["cat"], mapping["dog"]);
    println!("mapped shape: {:?}", encoded.dim());
    Ok(())
}
