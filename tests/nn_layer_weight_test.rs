#![cfg(feature = "neural_network")]

use rustyml::neural_network::layer::activation_layer::relu::ReLU;
use rustyml::neural_network::layer::activation_layer::tanh::Tanh;
use rustyml::neural_network::layer::dense::Dense;
use rustyml::neural_network::layer::layer_weight::LayerWeight;
use rustyml::neural_network::layer::recurrent_layer::simple_rnn::SimpleRNN;
use rustyml::neural_network::loss_function::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::optimizer::sgd::SGD;
use rustyml::neural_network::sequential::Sequential;

#[test]
fn test_get_weights() {
    // Create and compile the model
    let mut model = Sequential::new();
    model
        .add(Dense::new(4, 3, ReLU::new()).unwrap())
        .add(SimpleRNN::new(3, 2, Tanh::new()).unwrap());
    model.compile(SGD::new(0.01).unwrap(), MeanSquaredError::new());

    // Get all layer weights
    let weights = model.get_weights();
    assert_eq!(weights.len(), 2);

    // First layer is Dense(4 -> 3): weight (4, 3), bias (1, 3). `match` (not a bare `if let`)
    // so a wrong variant fails the test instead of silently passing.
    match &weights[0] {
        LayerWeight::Dense(dense_weights) => {
            assert_eq!(dense_weights.weight.shape(), &[4, 3]);
            assert_eq!(dense_weights.bias.shape(), &[1, 3]);
        }
        _ => panic!("expected Dense weights for layer 0"),
    }

    // Second layer is SimpleRNN(3 -> 2): kernel (3, 2), recurrent (2, 2), bias (1, 2).
    match &weights[1] {
        LayerWeight::SimpleRNN(rnn_weights) => {
            assert_eq!(rnn_weights.kernel.shape(), &[3, 2]);
            assert_eq!(rnn_weights.recurrent_kernel.shape(), &[2, 2]);
            assert_eq!(rnn_weights.bias.shape(), &[1, 2]);
        }
        _ => panic!("expected SimpleRNN weights for layer 1"),
    }
}
