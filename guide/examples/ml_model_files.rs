use ndarray::array;
use rustyml::error::Error;
use rustyml::prelude::machine_learning::LinearRegression;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[0.0], [1.0], [2.0]];
    let y = array![1.0, 3.0, 5.0];
    let mut model = LinearRegression::new(true);
    assert!(matches!(model.predict(&x), Err(Error::NotFitted(_))));
    model.fit(&x, &y)?;

    let path = std::env::temp_dir().join(format!("rustyml-guide-{}.bin", std::process::id()));
    let filename = path.to_str().ok_or("temporary path is not UTF-8")?;
    model.save_to_path(filename)?;
    let restored = LinearRegression::load_from_path(filename)?;
    let predictions = restored.predict(&array![[3.0]])?;
    println!("restored prediction: {:.1}", predictions[0]);
    std::fs::remove_file(path)?;
    Ok(())
}
