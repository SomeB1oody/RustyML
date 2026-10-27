use ndarray::array;
use rustyml::prelude::neural_network::*;

fn build_model() -> Result<Sequential, rustyml::error::Error> {
    SequentialBuilder::new_with_seed(42)
        .add(Dense::new(1, Activation::Linear)?.with_random_state(7))
        .build(&Shape::known(&[4, 1]))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let x = array![[-1.0_f32], [0.0], [1.0], [2.0]].into_dyn();
    let y = array![[-1.0_f32], [1.0], [3.0], [5.0]].into_dyn();
    let mut model = build_model()?;
    model.compile(SGD::new(0.05, 0.0, false, 0.0)?, MeanSquaredError::new());
    let initial_loss = model.evaluate(&x, &y)?;
    let history = model.fit_with_batches(&x, &y, 200, 2)?;
    let final_loss = model.evaluate(&x, &y)?;
    assert!(final_loss < initial_loss);
    assert!(final_loss < 0.0001);
    println!("epochs: {}", history.loss().len());
    println!("loss decreased: {}", final_loss < initial_loss);

    let path = std::env::temp_dir().join(format!("rustyml-nn-{}.bin", std::process::id()));
    model.save_to_path(&path)?;
    let mut restored = build_model()?;
    restored.load_from_path(&path)?;
    let prediction = restored.predict(&x)?;
    assert_eq!(model.predict(&x)?, prediction);
    println!("prediction shape: {:?}", prediction.shape());
    println!("restored predictions match: true");
    std::fs::remove_file(path)?;
    Ok(())
}
