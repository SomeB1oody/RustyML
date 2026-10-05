//! The roster of every layer type, for the tests that hold every layer to 1 contract
//!
//! No item of the crate lists its layer types, so this file is the only roster. Each case
//! gives a constructor, the input shapes, and the output shape that the layer gives for them.
//! [`visit_every_layer`] hands each case to a [`LayerVisitor`]. The visitor gets the concrete
//! layer type, so it can add the layer to a `SequentialBuilder` or box it.
//!
//! For each new layer type, add a case here and change [`LAYER_TYPE_COUNT`] in the same commit.

use ndarray::{ArrayD, IxDyn};
use rustyml::neural_network::layers::convolution::PaddingType;
use rustyml::neural_network::layers::*;
use rustyml::neural_network::traits::Layer;
use rustyml::neural_network::{Shape, Tensor};

/// The number of layer types in [`visit_every_layer`]
///
/// The tests that walk the roster compare their count of distinct layer types to this number.
/// A layer type that the roster skips, or holds twice, therefore fails each of these tests
pub const LAYER_TYPE_COUNT: usize = 74;

/// The values that a sample input holds
#[derive(Clone, Copy, Debug)]
pub enum Values {
    /// Real values in [-1, 1], with no 2 neighbors equal
    Real,
    /// Whole numbers in [0, n), for a layer that reads its input as indices
    Index(usize),
}

/// 1 roster case: the input shapes a layer takes, and the output shape it gives for them
#[derive(Clone, Debug)]
pub struct Case {
    /// 1 shape for each input of the layer. The batch axis is free
    pub inputs: Vec<Shape>,
    /// The output shape for `inputs`, as `Shape` displays it
    pub output: &'static str,
    /// The values of the sample input
    pub values: Values,
}

impl Case {
    /// The sample inputs of the case, with a batch of 2
    ///
    /// The values are a fixed function of the element index. 2 calls therefore give equal
    /// tensors
    pub fn sample(&self) -> Vec<Tensor> {
        self.inputs
            .iter()
            .enumerate()
            .map(|(input, shape)| {
                let dims: Vec<usize> = shape
                    .axes()
                    .iter()
                    .map(|extent| extent.unwrap_or(2))
                    .collect();
                let len: usize = dims.iter().product();
                let values = (0..len)
                    .map(|i| {
                        let i = i + 5 * input;
                        match self.values {
                            Values::Real => ((i * 37 + 11) % 23) as f32 / 11.0 - 1.0,
                            Values::Index(n) => ((i * 7 + 3) % n) as f32,
                        }
                    })
                    .collect();
                ArrayD::from_shape_vec(IxDyn(&dims), values).unwrap()
            })
            .collect()
    }

    /// A gradient of ones in the shape of the output, with a batch of 2
    pub fn output_grad(&self) -> Tensor {
        let dims: Vec<usize> = self
            .output
            .trim_matches(|c| c == '(' || c == ')')
            .split(", ")
            .map(|extent| extent.parse().unwrap_or(2))
            .collect();
        ArrayD::ones(IxDyn(&dims))
    }
}

/// A test that receives each roster case with its concrete layer type
pub trait LayerVisitor {
    /// Receives 1 case
    ///
    /// # Parameters
    ///
    /// - `case` - The input shapes, the output shape, and the sample values
    /// - `make` - Makes a new unbuilt layer of the case. 2 calls give 2 independent layers
    fn visit<L: Layer + 'static>(&mut self, case: Case, make: &dyn Fn() -> L);
}

/// Hands every roster case to `visitor`, in a fixed order
pub fn visit_every_layer(visitor: &mut impl LayerVisitor) {
    let flat = Shape::with_free_batch(&[1, 4]);
    let sequence = Shape::with_free_batch(&[1, 5, 2]);
    let signal = Shape::with_free_batch(&[1, 8, 2]);
    let image = Shape::with_free_batch(&[1, 8, 8, 2]);
    let volume = Shape::with_free_batch(&[1, 4, 4, 4, 2]);
    let unary = |input: &Shape, output: &'static str| Case {
        inputs: vec![input.clone()],
        output,
        values: Values::Real,
    };
    let pair = |output: &'static str| Case {
        inputs: vec![flat.clone(), flat.clone()],
        output,
        values: Values::Real,
    };

    // Dense and the shape layers
    visitor.visit(unary(&flat, "(None, 4)"), &|| {
        Dense::new(4, Linear::new()).unwrap()
    });
    visitor.visit(
        unary(&Shape::with_free_batch(&[1, 2, 3]), "(None, 6)"),
        &Flatten::new,
    );
    visitor.visit(unary(&flat, "(None, 4)"), &Identity::new);
    visitor.visit(unary(&flat, "(None, 2, 2)"), &|| {
        Reshape::new(vec![2, 2]).unwrap()
    });
    visitor.visit(
        unary(&Shape::with_free_batch(&[1, 2, 3]), "(None, 3, 2)"),
        &|| Permute::new(vec![2, 1]).unwrap(),
    );
    visitor.visit(unary(&flat, "(None, 3, 4)"), &|| {
        RepeatVector::new(3).unwrap()
    });
    visitor.visit(unary(&flat, "(None, 4)"), &|| Rescaling::new(2.0));
    visitor.visit(unary(&sequence, "(None, 5, 2)"), &|| Reverse::new(1));
    visitor.visit(
        Case {
            inputs: vec![Shape::with_free_batch(&[1, 5])],
            output: "(None, 5, 3)",
            values: Values::Index(10),
        },
        &|| Embedding::new(10, 3).unwrap(),
    );

    // Activations
    visitor.visit(unary(&flat, "(None, 4)"), &ReLU::new);
    visitor.visit(unary(&flat, "(None, 4)"), &|| LeakyReLU::new(0.1).unwrap());
    visitor.visit(unary(&flat, "(None, 4)"), &|| ELU::new(1.0).unwrap());
    visitor.visit(unary(&flat, "(None, 4)"), &SELU::new);
    visitor.visit(unary(&flat, "(None, 4)"), &Softplus::new);
    visitor.visit(unary(&flat, "(None, 4)"), &Softsign::new);
    visitor.visit(unary(&flat, "(None, 4)"), &HardSigmoid::new);
    visitor.visit(unary(&flat, "(None, 4)"), &Exponential::new);
    visitor.visit(unary(&flat, "(None, 4)"), &Linear::new);
    visitor.visit(unary(&flat, "(None, 4)"), &Sigmoid::new);
    visitor.visit(unary(&flat, "(None, 4)"), &Tanh::new);
    visitor.visit(unary(&flat, "(None, 4)"), &Softmax::new);
    visitor.visit(unary(&flat, "(None, 4)"), &|| PReLU::new(0.25).unwrap());

    // Convolution
    visitor.visit(unary(&signal, "(None, 6, 4)"), &|| {
        Conv1D::new(4, 3, 1, ReLU::new()).unwrap()
    });
    visitor.visit(unary(&image, "(None, 6, 6, 4)"), &|| {
        Conv2D::new(4, (3, 3), (1, 1), ReLU::new()).unwrap()
    });
    visitor.visit(unary(&volume, "(None, 3, 3, 3, 3)"), &|| {
        Conv3D::new(3, (2, 2, 2), (1, 1, 1), ReLU::new()).unwrap()
    });
    visitor.visit(
        unary(&Shape::with_free_batch(&[1, 4, 2]), "(None, 8, 3)"),
        &|| Conv1DTranspose::new(3, 2, 2, Linear::new()).unwrap(),
    );
    visitor.visit(
        unary(&Shape::with_free_batch(&[1, 4, 4, 2]), "(None, 8, 8, 3)"),
        &|| Conv2DTranspose::new(3, (2, 2), (2, 2), Linear::new()).unwrap(),
    );
    visitor.visit(
        unary(
            &Shape::with_free_batch(&[1, 2, 2, 2, 1]),
            "(None, 4, 4, 4, 2)",
        ),
        &|| Conv3DTranspose::new(2, (2, 2, 2), (2, 2, 2), Linear::new()).unwrap(),
    );
    // A depth multiplier of 2 gives the depthwise kernel another shape than the default
    visitor.visit(unary(&signal, "(None, 8, 4)"), &|| {
        SeparableConv1D::new(4, 3, 1, 2, ReLU::new())
            .unwrap()
            .with_padding(PaddingType::Same)
    });
    visitor.visit(unary(&image, "(None, 6, 6, 4)"), &|| {
        SeparableConv2D::new(4, (3, 3), (1, 1), 1, ReLU::new()).unwrap()
    });
    // A depthwise layer gives the input channel count times the depth multiplier
    visitor.visit(unary(&signal, "(None, 6, 4)"), &|| {
        DepthwiseConv1D::new(3, 1, ReLU::new())
            .unwrap()
            .with_depth_multiplier(2)
            .unwrap()
    });
    visitor.visit(unary(&image, "(None, 6, 6, 2)"), &|| {
        DepthwiseConv2D::new((3, 3), (1, 1), ReLU::new()).unwrap()
    });

    // Pooling
    visitor.visit(unary(&signal, "(None, 4, 2)"), &|| MaxPooling1D::new(2));
    visitor.visit(unary(&image, "(None, 4, 4, 2)"), &|| {
        MaxPooling2D::new((2, 2))
    });
    visitor.visit(unary(&volume, "(None, 2, 2, 2, 2)"), &|| {
        MaxPooling3D::new((2, 2, 2))
    });
    visitor.visit(unary(&signal, "(None, 4, 2)"), &|| AveragePooling1D::new(2));
    visitor.visit(unary(&image, "(None, 4, 4, 2)"), &|| {
        AveragePooling2D::new((2, 2))
    });
    visitor.visit(unary(&volume, "(None, 2, 2, 2, 2)"), &|| {
        AveragePooling3D::new((2, 2, 2))
    });
    visitor.visit(unary(&signal, "(None, 2)"), &GlobalMaxPooling1D::new);
    visitor.visit(unary(&image, "(None, 2)"), &GlobalMaxPooling2D::new);
    visitor.visit(unary(&volume, "(None, 2)"), &GlobalMaxPooling3D::new);
    visitor.visit(unary(&signal, "(None, 2)"), &GlobalAveragePooling1D::new);
    visitor.visit(unary(&image, "(None, 2)"), &GlobalAveragePooling2D::new);
    visitor.visit(unary(&volume, "(None, 2)"), &GlobalAveragePooling3D::new);

    // Resampling and borders
    visitor.visit(
        unary(&Shape::with_free_batch(&[1, 4, 2]), "(None, 8, 2)"),
        &|| UpSampling1D::new(2).unwrap(),
    );
    visitor.visit(
        unary(&Shape::with_free_batch(&[1, 4, 4, 2]), "(None, 8, 8, 2)"),
        &|| UpSampling2D::new(2, Interpolation::Nearest).unwrap(),
    );
    visitor.visit(
        unary(
            &Shape::with_free_batch(&[1, 2, 2, 2, 1]),
            "(None, 4, 4, 4, 1)",
        ),
        &|| UpSampling3D::new(2).unwrap(),
    );
    visitor.visit(
        unary(&Shape::with_free_batch(&[1, 4, 2]), "(None, 6, 2)"),
        &|| ZeroPadding1D::new(1),
    );
    visitor.visit(
        unary(&Shape::with_free_batch(&[1, 4, 4, 2]), "(None, 6, 6, 2)"),
        &|| ZeroPadding2D::new(1),
    );
    visitor.visit(
        unary(
            &Shape::with_free_batch(&[1, 2, 2, 2, 1]),
            "(None, 4, 4, 4, 1)",
        ),
        &|| ZeroPadding3D::new(1),
    );
    visitor.visit(
        unary(&Shape::with_free_batch(&[1, 6, 2]), "(None, 4, 2)"),
        &|| Cropping1D::new(1),
    );
    visitor.visit(
        unary(&Shape::with_free_batch(&[1, 6, 6, 2]), "(None, 4, 4, 2)"),
        &|| Cropping2D::new(1),
    );
    visitor.visit(unary(&volume, "(None, 2, 2, 2, 2)"), &|| Cropping3D::new(1));

    // Recurrent
    visitor.visit(unary(&sequence, "(None, 3)"), &|| {
        SimpleRNN::new(3, Tanh::new()).unwrap()
    });
    visitor.visit(unary(&sequence, "(None, 3)"), &|| {
        LSTM::new(3, Tanh::new()).unwrap()
    });
    visitor.visit(unary(&sequence, "(None, 3)"), &|| {
        GRU::new(3, Tanh::new()).unwrap()
    });

    // Regularization
    visitor.visit(unary(&flat, "(None, 4)"), &|| Dropout::new(0.5).unwrap());
    visitor.visit(unary(&signal, "(None, 8, 2)"), &|| {
        SpatialDropout1D::new(0.5).unwrap()
    });
    visitor.visit(unary(&image, "(None, 8, 8, 2)"), &|| {
        SpatialDropout2D::new(0.5).unwrap()
    });
    visitor.visit(unary(&volume, "(None, 4, 4, 4, 2)"), &|| {
        SpatialDropout3D::new(0.5).unwrap()
    });
    visitor.visit(unary(&flat, "(None, 4)"), &|| {
        GaussianDropout::new(0.3).unwrap()
    });
    visitor.visit(unary(&flat, "(None, 4)"), &|| {
        GaussianNoise::new(0.1).unwrap()
    });
    visitor.visit(unary(&flat, "(None, 4)"), &|| {
        BatchNormalization::new(0.9, 1e-5).unwrap()
    });
    visitor.visit(unary(&flat, "(None, 4)"), &|| {
        LayerNormalization::new(1e-5).unwrap()
    });
    visitor.visit(unary(&signal, "(None, 8, 2)"), &|| {
        GroupNormalization::new(2, 1e-5).unwrap()
    });
    visitor.visit(unary(&signal, "(None, 8, 2)"), &|| {
        InstanceNormalization::new(1e-5).unwrap()
    });
    visitor.visit(unary(&flat, "(None, 4)"), &|| {
        UnitNormalization::new(UnitNormalizationAxis::Default).unwrap()
    });

    // Merge
    visitor.visit(pair("(None, 4)"), &Add::new);
    visitor.visit(pair("(None, 4)"), &Subtract::new);
    visitor.visit(pair("(None, 4)"), &Multiply::new);
    visitor.visit(pair("(None, 4)"), &Average::new);
    visitor.visit(pair("(None, 4)"), &Maximum::new);
    visitor.visit(pair("(None, 4)"), &Minimum::new);
    visitor.visit(pair("(None, 8)"), &|| Concatenate::new(-1));
}
