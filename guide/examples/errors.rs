use ndarray::array;
use rustyml::error::{Error, RustymlResult};
use rustyml::machine_learning::LinearRegression;

fn main() -> RustymlResult<()> {
    let x = array![[1.0], [2.0]];
    let mut model = LinearRegression::new(true);
    match model.predict(&x) {
        Err(Error::NotFitted(_)) => println!("fit the model before prediction"),
        Err(error) => return Err(error),
        Ok(_) => println!("prediction succeeded"),
    }
    model.fit(&x, &array![3.0, 5.0])?;
    println!("prediction: {:.1}", model.predict(&array![[3.0]])?[0]);
    Ok(())
}
