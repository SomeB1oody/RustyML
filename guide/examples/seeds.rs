use ndarray::array;
use rustyml::error::RustymlResult;
use rustyml::utils::train_test_split::train_test_split;
use rustyml::{clear_global_seed, set_global_seed};

fn main() -> RustymlResult<()> {
    let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    set_global_seed(42);
    let first = train_test_split(x.clone(), y.clone(), Some(0.33), None)?;
    set_global_seed(42);
    let second = train_test_split(x, y, Some(0.33), None)?;
    println!("same test rows: {}", first.1 == second.1);
    clear_global_seed();
    Ok(())
}
