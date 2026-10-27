use ndarray::array;
use rustyml::prelude::machine_learning::{
    LeastSquaresSolver, LinearRegression, LogisticRegression, RegularizationType,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[-2.0], [-1.0], [1.0], [2.0]];
    let y = array![0.0, 0.0, 1.0, 1.0];
    let mut classifier = LogisticRegression::new(true, 0.1, 10_000, 1e-8)?
        .with_regularization(RegularizationType::L2(0.1))?;
    classifier.fit(&x, &y)?;
    let queries = array![[-1.5], [1.5]];
    let probabilities = classifier.predict_proba(&queries)?;
    println!(
        "positive probabilities: {:.3}, {:.3}",
        probabilities[0], probabilities[1]
    );
    println!(
        "binary labels: {:?}",
        classifier.predict(&queries)?.to_vec()
    );

    let targets = array![-3.0, -1.0, 3.0, 5.0];
    let mut regression = LinearRegression::new(true)
        .with_solver(LeastSquaresSolver::GradientDescent {
            learning_rate: 0.05,
            max_iter: 10_000,
            tol: 1e-8,
        })?
        .with_regularization(RegularizationType::L1(0.01))?;
    regression.fit(&x, &targets)?;
    println!("L1 predictions: {}", regression.predict(&queries)?.len());
    Ok(())
}
