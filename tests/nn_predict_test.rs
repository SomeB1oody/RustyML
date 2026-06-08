#![cfg(feature = "neural_network")]

//! Verifies the `Layer::predict` (inference, `&self`, no-cache) path against `forward` in eval mode.
//!
//! For every layer, `predict(x)` must equal what `forward(x)` produces in inference mode — it only
//! skips the backward caches (and, for mode-dependent layers, always uses inference behavior). This
//! guards the per-layer `predict` implementations added for the `&self` inference API.

use approx::assert_abs_diff_eq;
use ndarray::IxDyn;
use rustyml::neural_network::Tensor;
use rustyml::neural_network::layers::activation::Activation;
use rustyml::neural_network::layers::activation::{relu::ReLU, softmax::Softmax};
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::convolution::conv_1d::Conv1D;
use rustyml::neural_network::layers::convolution::conv_2d::Conv2D;
use rustyml::neural_network::layers::convolution::conv_3d::Conv3D;
use rustyml::neural_network::layers::convolution::depthwise_conv_2d::DepthwiseConv2D;
use rustyml::neural_network::layers::convolution::separable_conv_2d::SeparableConv2D;
use rustyml::neural_network::layers::dense::Dense;
use rustyml::neural_network::layers::flatten::Flatten;
use rustyml::neural_network::layers::pooling::average_pooling_1d::AveragePooling1D;
use rustyml::neural_network::layers::pooling::global_average_pooling_2d::GlobalAveragePooling2D;
use rustyml::neural_network::layers::pooling::global_max_pooling_2d::GlobalMaxPooling2D;
use rustyml::neural_network::layers::pooling::max_pooling_2d::MaxPooling2D;
use rustyml::neural_network::layers::recurrent::gru::GRU;
use rustyml::neural_network::layers::recurrent::lstm::LSTM;
use rustyml::neural_network::layers::recurrent::simple_rnn::SimpleRNN;
use rustyml::neural_network::layers::regularization::dropout::dropout::Dropout;
use rustyml::neural_network::layers::regularization::dropout::spatial_dropout_2d::SpatialDropout2D;
use rustyml::neural_network::layers::regularization::noise_injection::gaussian_noise::GaussianNoise;
use rustyml::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization;
use rustyml::neural_network::layers::regularization::normalization::group_normalization::GroupNormalization;
use rustyml::neural_network::layers::regularization::normalization::instance_normalization::InstanceNormalization;
use rustyml::neural_network::layers::regularization::normalization::layer_normalization::{
    LayerNormalization, LayerNormalizationAxis,
};
use rustyml::neural_network::traits::Layer;

/// A deterministic, varying input of the given shape.
fn ramp(shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    let data: Vec<f32> = (0..n).map(|i| 0.13 * (i as f32) - 0.5).collect();
    Tensor::from_shape_vec(IxDyn(shape), data).unwrap()
}

/// Asserts `layer.predict(x)` equals `layer.forward(x)` once the layer is in inference mode.
fn assert_predict_eq_eval_forward(layer: &mut dyn Layer, shape: &[usize], name: &str) {
    let x = ramp(shape);
    // Put mode-dependent layers (dropout, batch norm, …) into inference mode so `forward` matches
    // `predict`'s always-inference behavior; a no-op for every other layer.
    layer.set_training_if_mode_dependent(false);
    let f = layer.forward(&x).unwrap();
    let p = layer.predict(&x).unwrap();
    assert_eq!(f.shape(), p.shape(), "{name}: predict/forward shape mismatch");
    for (a, b) in f.iter().zip(p.iter()) {
        assert_abs_diff_eq!(*a, *b, epsilon = 1e-4);
    }
}

#[test]
fn predict_matches_forward_dense_conv_pool_flatten_activation() {
    assert_predict_eq_eval_forward(
        &mut Dense::new(3, 2, Activation::Linear).unwrap(),
        &[4, 3],
        "Dense",
    );
    assert_predict_eq_eval_forward(
        &mut Conv1D::new(2, 2, vec![1, 1, 5], 1, PaddingType::Valid, Activation::Linear).unwrap(),
        &[1, 1, 5],
        "Conv1D",
    );
    assert_predict_eq_eval_forward(
        &mut Conv2D::new(
            2,
            (2, 2),
            vec![1, 1, 4, 4],
            (1, 1),
            PaddingType::Same,
            Activation::Linear,
        )
        .unwrap(),
        &[1, 1, 4, 4],
        "Conv2D(Same)",
    );
    assert_predict_eq_eval_forward(
        &mut Conv3D::new(
            2,
            (2, 2, 2),
            vec![1, 1, 3, 3, 3],
            (1, 1, 1),
            PaddingType::Valid,
            Activation::Linear,
        )
        .unwrap(),
        &[1, 1, 3, 3, 3],
        "Conv3D",
    );
    assert_predict_eq_eval_forward(
        &mut SeparableConv2D::new(
            2,
            (2, 2),
            vec![1, 2, 4, 4],
            (1, 1),
            PaddingType::Valid,
            1,
            Activation::Linear,
        )
        .unwrap(),
        &[1, 2, 4, 4],
        "SeparableConv2D",
    );
    let mut dw = DepthwiseConv2D::new(
        2,
        (2, 2),
        vec![1, 2, 4, 4],
        (1, 1),
        PaddingType::Valid,
        Activation::Linear,
    )
    .unwrap();
    assert_predict_eq_eval_forward(&mut dw, &[1, 2, 4, 4], "DepthwiseConv2D");
    assert_predict_eq_eval_forward(
        &mut MaxPooling2D::new((2, 2), vec![1, 1, 4, 4], None).unwrap(),
        &[1, 1, 4, 4],
        "MaxPooling2D",
    );
    assert_predict_eq_eval_forward(
        &mut GlobalMaxPooling2D::new(),
        &[1, 2, 4, 4],
        "GlobalMaxPooling2D",
    );
    assert_predict_eq_eval_forward(
        &mut GlobalAveragePooling2D::new(),
        &[1, 2, 4, 4],
        "GlobalAveragePooling2D",
    );
    assert_predict_eq_eval_forward(
        &mut AveragePooling1D::new(2, vec![1, 1, 8], Some(2)).unwrap(),
        &[1, 1, 8],
        "AveragePooling1D",
    );
    assert_predict_eq_eval_forward(&mut Flatten::new(vec![1, 2, 3]).unwrap(), &[1, 2, 3], "Flatten");
    assert_predict_eq_eval_forward(&mut ReLU::new(), &[2, 3], "ReLU");
    assert_predict_eq_eval_forward(&mut Softmax::new(), &[2, 3], "Softmax");
}

#[test]
fn predict_matches_forward_recurrent() {
    // The recurrent predicts must replicate the per-timestep recurrence (without BPTT caches).
    assert_predict_eq_eval_forward(
        &mut SimpleRNN::new(2, 3, Activation::Tanh).unwrap(),
        &[2, 4, 2],
        "SimpleRNN",
    );
    assert_predict_eq_eval_forward(&mut LSTM::new(2, 3, Activation::Tanh).unwrap(), &[2, 4, 2], "LSTM");
    assert_predict_eq_eval_forward(&mut GRU::new(2, 3, Activation::Tanh).unwrap(), &[2, 4, 2], "GRU");
}

#[test]
fn predict_matches_forward_mode_dependent() {
    // Dropout / noise are identity at inference; norm layers use their inference path.
    assert_predict_eq_eval_forward(&mut Dropout::new(0.5, vec![2, 3]).unwrap(), &[2, 3], "Dropout");
    assert_predict_eq_eval_forward(
        &mut SpatialDropout2D::new(0.5, vec![1, 2, 4, 4]).unwrap(),
        &[1, 2, 4, 4],
        "SpatialDropout2D",
    );
    assert_predict_eq_eval_forward(
        &mut GaussianNoise::new(0.1, vec![2, 3]).unwrap(),
        &[2, 3],
        "GaussianNoise",
    );
    assert_predict_eq_eval_forward(
        &mut BatchNormalization::new(vec![4, 3], 0.9, 1e-5).unwrap(),
        &[4, 3],
        "BatchNormalization",
    );
    assert_predict_eq_eval_forward(
        &mut LayerNormalization::new(vec![4, 3], LayerNormalizationAxis::Default, 1e-5).unwrap(),
        &[4, 3],
        "LayerNormalization",
    );
    assert_predict_eq_eval_forward(
        &mut InstanceNormalization::new(vec![2, 3, 4], 1, 1e-5).unwrap(),
        &[2, 3, 4],
        "InstanceNormalization",
    );
    assert_predict_eq_eval_forward(
        &mut GroupNormalization::new(vec![2, 4, 4], 2, 1, 1e-5).unwrap(),
        &[2, 4, 4],
        "GroupNormalization",
    );
}
