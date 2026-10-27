use ndarray::array;
use rustyml::prelude::machine_learning::{KNN, LinearRegression};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[0.0], [1.0], [2.0], [3.0]];
    let y = array![1.0, 3.0, 5.0, 7.0];
    let mut regression = LinearRegression::new(true);
    regression.fit(&x, &y)?;
    let predictions = regression.predict(&array![[4.0], [5.0]])?;
    println!("regression: {:.1}, {:.1}", predictions[0], predictions[1]);

    let labels = array![0_i32, 0, 1, 1];
    let mut classifier = KNN::<i32>::new(1)?;
    classifier.fit(&x, &labels)?;
    let predictions = classifier.predict(&array![[0.1], [2.9]])?;
    println!("classes: {:?}", predictions.to_vec());
    Ok(())
}
