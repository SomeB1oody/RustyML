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

    // Examine the weights of the first layer (Dense layer)
    if let LayerWeight::Dense(dense_weights) = &weights[0] {
        println!("Dense layer weights: {:?}", dense_weights.weight);
        println!("Dense layer bias: {:?}", dense_weights.bias);
    }

    // Examine the weights of the second layer (SimpleRNN layer)
    if let LayerWeight::SimpleRNN(rnn_weights) = &weights[1] {
        println!("SimpleRNN layer input weights: {:?}", rnn_weights.kernel);
        println!(
            "SimpleRNN layer recurrent weights: {:?}",
            rnn_weights.recurrent_kernel
        );
        println!("SimpleRNN layer bias: {:?}", rnn_weights.bias);
    }
}
