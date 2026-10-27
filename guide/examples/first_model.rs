use ndarray::array;
use rustyml::error::RustymlResult;
use rustyml::machine_learning::LinearRegression;
use rustyml::metrics::mean_absolute_error;
use rustyml::utils::train_test_split::train_test_split;

fn main() -> RustymlResult<()> {
    let x = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
    let y = array![1.0, 3.0, 5.0, 7.0, 9.0, 11.0];
    let (x_train, x_test, y_train, y_test) = train_test_split(x, y, Some(0.33), Some(42))?;

    let mut model = LinearRegression::new(true);
    model.fit(&x_train, &y_train)?;
    let predictions = model.predict(&x_test)?;
    println!("training rows: {}", x_train.nrows());
    println!("test rows: {}", x_test.nrows());
    println!(
        "test MAE: {:.6}",
        mean_absolute_error(&y_test, &predictions)
    );

    let query = array![[6.0]];
    println!("prediction for 6: {:.3}", model.predict(&query)?[0]);
    Ok(())
}
