use ndarray::Array;
use rustyml::prelude::neural_network::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let images = Array::<f32, _>::zeros((2, 8, 8, 1)).into_dyn();
    let cnn = SequentialBuilder::new()
        .add(Conv2D::new(4, (3, 3), (1, 1), Activation::ReLU)?.with_random_state(11))
        .add(MaxPooling2D::new((2, 2)))
        .add(Flatten::new())
        .add(Dense::new(3, Activation::Linear)?.with_random_state(12))
        .build(&Shape::known(images.shape()))?;
    println!("image predictions: {:?}", cnn.predict(&images)?.shape());

    let sequence = Array::<f32, _>::zeros((2, 5, 3)).into_dyn();
    let rnn = SequentialBuilder::new()
        .add(
            LSTM::new(4, Activation::Tanh)?
                .with_random_state(13)
                .with_return_sequences(true),
        )
        .add(Dense::new(2, Activation::Linear)?.with_random_state(14))
        .build(&Shape::known(sequence.shape()))?;
    println!(
        "sequence predictions: {:?}",
        rnn.predict(&sequence)?.shape()
    );
    Ok(())
}
