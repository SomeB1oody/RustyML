use crate::error::ModelError;
use crate::neural_network::*;
use approx::{assert_abs_diff_eq, assert_relative_eq};
use ndarray::prelude::*;

mod activation_test;
mod ada_grad_test;
mod average_pooling_1d_test;
mod average_pooling_2d_test;
mod average_pooling_3d_test;
mod batch_normalization_test;
mod conv1d_test;
mod conv2d_test;
mod conv3d_test;
mod dense_test;
mod depthwise_conv2d_test;
mod dropout_test;
mod flatten_test;
mod global_average_pooling_1d_test;
mod global_average_pooling_2d_test;
mod global_average_pooling_3d_test;
mod global_max_pooling_1d_test;
mod global_max_pooling_2d_test;
mod global_max_pooling_3d_test;
mod gru_test;
mod layer_weight_test;
mod loss_function_test;
mod lstm_test;
mod max_pooling_1d_test;
mod max_pooling_2d_test;
mod max_pooling_3d_test;
mod optimizer_test;
mod separable_conv2d_test;
mod sequential_test;
mod simple_rnn_test;
mod spatial_dropout_1d_test;
mod spatial_dropout_2d_test;
mod spatial_dropout_3d_test;

fn generate_data(batch_size: usize, channels: usize, length: usize) -> Tensor {
    let mut input_data = Tensor::zeros(IxDyn(&[batch_size, channels, length]));

    // Initialize input data, set specific values for testing average calculation
    for b in 0..batch_size {
        for c in 0..channels {
            for l in 0..length {
                input_data[[b, c, l]] = (b * 100 + c * 10 + l) as f32;
            }
        }
    }

    input_data
}
