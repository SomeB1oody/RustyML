use ndarray::array;
use rustyml::utils::{StandardScaler, train_test_split, train_test_split_stratified};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![
        [1.0, 10.0],
        [2.0, 20.0],
        [3.0, 30.0],
        [4.0, 40.0],
        [5.0, 50.0],
        [6.0, 60.0]
    ];
    let y = array![0, 0, 0, 1, 1, 1];
    let (x_train, x_test, y_train, y_test) =
        train_test_split(x.clone(), y.clone(), Some(0.33), Some(42))?;
    println!("train: {:?}, labels: {}", x_train.dim(), y_train.len());
    println!("test: {:?}, labels: {}", x_test.dim(), y_test.len());
    let mut scaler = StandardScaler::new();
    scaler.fit(&x_train)?;
    let z_train = scaler.transform(&x_train)?;
    let z_test = scaler.transform(&x_test)?;
    println!("scaled: {:?}, {:?}", z_train.dim(), z_test.dim());
    let (_, _, strat_train, strat_test) = train_test_split_stratified(x, y, Some(0.33), Some(42))?;
    println!(
        "class 0 counts: train {}, test {}",
        strat_train.iter().filter(|&&v| v == 0).count(),
        strat_test.iter().filter(|&&v| v == 0).count()
    );
    Ok(())
}
