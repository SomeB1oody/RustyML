//! PReLU activation layer, whose negative-side slope is a trainable parameter, and the
//! shared-axes rule that decides how many slopes it holds

use crate::error::Error;
use crate::neural_network::layers::ParamCounts;
use crate::neural_network::layers::validation::{start_build, validate_weight_shape};
use crate::neural_network::layers::{built_layer_shape_functions, named_weight_layer_functions};
use crate::neural_network::traits::{LayerBase, ParamRef, UnaryLayer};
use crate::neural_network::{Ctx, Shape, Tensor};
use crate::parallel_gates::cheap_map_parallel_threshold;
use ndarray::{ArrayD, Axis, Zip};

/// Parametric ReLU activation layer, which learns its negative-side slope
///
/// Applies `f(x) = x` for `x >= 0`, and `f(x) = alpha * x` below 0. The transform is elementwise
/// and keeps the input shape. Unlike
/// [`LeakyReLU`](crate::neural_network::layers::activation::leaky_relu::LeakyReLU), which holds
/// a constant `negative_slope`, `alpha` here is a trainable parameter. The layer learns how much
/// of the negative side to keep, and the optimizer updates it with the rest of the model.
///
/// The layer holds 1 slope per position of the input shape with the batch axis removed. An
/// input of shape `[batch, 4, 5, 6]` therefore carries `4 * 5 * 6` slopes.
/// [`PReLU::with_shared_axes`] makes the named axes share 1 slope. This gives 1 slope per
/// channel instead of 1 per pixel in a convolutional stack.
///
/// # Notes
///
/// A shared axis accepts any extent at forward time, because its slope broadcasts. Every other
/// axis after the batch axis must match the configured `input_shape`. The layer never checks
/// the batch axis itself, so it takes a batch of any size.
///
/// The derivative at exactly 0 is 0, which is neither branch. An `alpha` of 0 therefore gives
/// the same forward transform and the same gradient as
/// [`ReLU`](crate::neural_network::layers::activation::relu::ReLU). This makes 0 the natural
/// starting value. `LeakyReLU` instead has a derivative of 1 at exactly 0.
///
/// Weight decay skips `alpha`, in every optimizer that takes a `weight_decay` argument. Decay
/// pulls a parameter toward 0. A slope of 0 turns this layer back into `ReLU`, which would
/// fight what the layer learns.
///
/// [`Activation`](crate::neural_network::layers::activation::Activation) has no PReLU variant,
/// and it cannot get one. A variant carries no state, and this layer carries a trainable array.
/// Place it after the layer whose output it activates, which is the split form that every
/// activation layer supports.
///
/// # Examples
///
/// ```rust
/// use ndarray::Array2;
/// use rustyml::neural_network::layers::{Activation, Dense, PReLU};
/// use rustyml::neural_network::losses::MeanSquaredError;
/// use rustyml::neural_network::optimizers::SGD;
/// use rustyml::neural_network::Ctx;
/// use rustyml::neural_network::Shape;
/// use rustyml::neural_network::sequential::SequentialBuilder;
/// use rustyml::neural_network::traits::UnaryLayer;
///
/// // 2 samples of 3 features, with negative values that exercise the learned branch
/// let x = Array2::from_shape_vec((2, 3), vec![-1.0, 2.0, -3.0, 4.0, -5.0, 6.0])
///     .unwrap()
///     .into_dyn();
/// let y = Array2::zeros((2, 1)).into_dyn();
///
/// // The layer alone starts from a slope of 0.25 on each of the 3 features
/// let mut slopes = PReLU::new(0.25).unwrap();
/// slopes.build(&Shape::known(x.shape())).unwrap();
/// let mut ctx = Ctx::inference();
/// let out = slopes.forward(&x, &mut ctx).unwrap();
/// assert_eq!(out[[0, 0]], -0.25);
/// assert_eq!(out[[0, 1]], 2.0);
///
/// // Inside a model the 3 slopes train together with the dense weights
/// let mut model = SequentialBuilder::new()
///     .add(Dense::new(3, Activation::Linear).unwrap())
///     .add(PReLU::new(0.25).unwrap())
///     .add(Dense::new(1, Activation::Linear).unwrap())
///     .build(&Shape::known(x.shape()))
///     .unwrap();
/// model.compile(SGD::new(0.01, 0.0, false, 0.0).unwrap(), MeanSquaredError::new());
///
/// model.fit(&x, &y, 2).unwrap();
/// assert_eq!(model.predict(&x).unwrap().shape(), &[2, 1]);
/// ```
///
/// # Performance
///
/// The forward pass and the elementwise half of the backward pass run in parallel once the
/// element count reaches the shared cheap-map gate. Override that gate through
/// [`crate::tuning::elementwise::set_cheap_map_f32`]. Both passes are elementwise, so the gate
/// never changes a value.
///
/// The slope gradient reduces over the batch axis and over every shared axis. That reduction
/// stays serial, in axis order, so it reproduces bit for bit across runs.
#[derive(Debug)]
pub struct PReLU {
    /// Shape of the input tensor, batch axis first
    input_shape: Vec<usize>,
    /// Axes after the batch axis that share 1 slope, in increasing order and without repeats
    shared_axes: Vec<usize>,
    /// Value that filled every slope at construction. `with_shared_axes` refills the resized
    /// array with it
    alpha_init: f32,
    /// Trainable negative-side slopes. Rank is 1 below the input rank, and a shared axis has
    /// extent 1
    alpha: ArrayD<f32>,
    /// Shape the layer was built for, batch axis first. `None` before the build
    built: Option<Shape>,
}

impl PReLU {
    /// Creates a new PReLU layer with every slope set to the same starting value
    ///
    /// # Parameters
    ///
    /// - `alpha` - Starting value of every negative-side slope. Use 0 to start from the
    ///   [`ReLU`](crate::neural_network::layers::activation::relu::ReLU) transform
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new `PReLU` layer, or an error if a parameter is unusable
    ///
    /// # Notes
    ///
    /// Without [`PReLU::with_shared_axes`], the layer holds 1 slope per position of the build
    /// shape with the batch axis removed
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If `alpha` is not finite
    pub fn new(alpha: f32) -> Result<Self, Error> {
        if !alpha.is_finite() {
            return Err(Error::invalid_parameter(
                "alpha",
                "must be finite, because a non-finite slope makes every negative element \
                 non-finite",
            ));
        }

        Ok(Self {
            input_shape: Vec::new(),
            shared_axes: Vec::new(),
            alpha_init: alpha,
            alpha: ArrayD::from_elem(Vec::new(), alpha),
            built: None,
        })
    }

    /// Fills the slope array at the extents the build settled
    ///
    /// Axis `d` of the input is axis `d - 1` of the slope array, because the batch axis is not
    /// part of it. A shared axis drops to extent 1, and the slope then broadcasts over it
    fn allocate_alpha(&mut self) {
        let mut param_shape = self.input_shape[1..].to_vec();
        for &axis in &self.shared_axes {
            param_shape[axis - 1] = 1;
        }
        self.alpha = ArrayD::from_elem(param_shape, self.alpha_init);
    }

    /// Makes the named axes share 1 slope, and resizes the slope array to match
    ///
    /// Each named axis drops to extent 1 in the slope array, and the slope then broadcasts back
    /// over that axis. A 4-D convolutional input of `[batch, height, width, channels]` with
    /// `shared_axes` of `[1, 2]` therefore holds 1 slope per channel. That is the usual choice
    /// for a convolutional stack, where a per-pixel slope both overfits and grows the parameter
    /// count with the image size.
    ///
    /// The layer also stops checking a shared axis at forward time, so it accepts any extent
    /// on it. The same layer then serves images of several sizes.
    ///
    /// # Parameters
    ///
    /// - `shared_axes` - Axes of the input that share 1 slope, counted from the batch axis at 0
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - The updated layer, or an error if an axis is unusable
    ///
    /// # Notes
    ///
    /// The resize refills every slope with the starting value that [`PReLU::new`] received.
    /// Call this before [`PReLU::set_weights`]
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If an axis is 0, which is the batch axis and is always
    ///   shared, or appears more than 1 time. [`UnaryLayer::build`] reports an axis that the
    ///   rank of the real input does not hold
    pub fn with_shared_axes(mut self, shared_axes: Vec<usize>) -> Result<Self, Error> {
        let mut sorted = shared_axes;
        sorted.sort_unstable();
        for (position, &axis) in sorted.iter().enumerate() {
            if axis == 0 {
                return Err(Error::invalid_parameter(
                    "shared_axes",
                    "holds axis 0, which is the batch axis. Every slope is already shared over \
                     the batch",
                ));
            }
            if position > 0 && sorted[position - 1] == axis {
                return Err(Error::invalid_parameter(
                    "shared_axes",
                    format!("holds axis {axis} more than 1 time"),
                ));
            }
        }

        self.shared_axes = sorted;
        if self.built.is_some() {
            self.allocate_alpha();
        }
        Ok(self)
    }

    /// Sets the negative-side slopes for this layer
    ///
    /// # Parameters
    ///
    /// - `alpha` - Slope array with the shape this layer holds, which is `input_shape` without
    ///   the batch axis and with extent 1 on every shared axis
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - Ok when `alpha` matches the layer's configured shape
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::NotBuilt)` - If the layer holds no array yet
    /// - `Error::NeuralNetwork(NnError::WeightShape)` - If `alpha` does not match the layer's
    ///   configured shape
    pub fn set_weights(&mut self, alpha: ArrayD<f32>) -> Result<(), Error> {
        if self.built.is_none() {
            return Err(Error::not_built("PReLU"));
        }
        validate_weight_shape("alpha", self.alpha.shape(), alpha.shape())?;

        self.alpha = alpha.as_standard_layout().into_owned();
        Ok(())
    }

    /// Checks the rank and every axis that does not share a slope
    ///
    /// # Errors
    ///
    /// - `Error::NeuralNetwork(NnError::NotBuilt)` - If the layer holds no array yet
    /// - `Error::EmptyInput` - If the input holds no element
    /// - `Error::InvalidInput` - If the rank differs from the configured rank, or an axis that
    ///   is not shared has an extent the slope array cannot cover
    fn validate_input(&self, input: &Tensor) -> Result<(), Error> {
        if self.built.is_none() {
            return Err(Error::not_built("PReLU"));
        }
        if input.is_empty() {
            return Err(Error::empty_input("input tensor"));
        }
        if input.ndim() != self.input_shape.len() {
            return Err(Error::invalid_input(format!(
                "PReLU layer expects an input of rank {}, got rank {}",
                self.input_shape.len(),
                input.ndim()
            )));
        }
        for axis in 1..self.input_shape.len() {
            if self.shared_axes.contains(&axis) {
                continue;
            }
            if input.shape()[axis] != self.input_shape[axis] {
                return Err(Error::invalid_input(format!(
                    "PReLU layer holds 1 slope per position of axis {axis}, so that axis must \
                     have extent {}, got {}. Add the axis to shared_axes to accept any extent",
                    self.input_shape[axis],
                    input.shape()[axis]
                )));
            }
        }
        Ok(())
    }

    /// Applies the transform. `forward` calls this on every pass, training and inference alike
    fn activate(&self, input: &Tensor) -> Result<Tensor, Error> {
        self.validate_input(input)?;

        let slopes = self
            .alpha
            .broadcast(input.raw_dim())
            .expect("validate_input accepts only a shape the slope array covers");
        let mut output = Tensor::zeros(input.raw_dim());
        let p_relu = |out: &mut f32, &x: &f32, &a: &f32| {
            *out = if x >= 0.0 { x } else { a * x };
        };
        if input.len() >= cheap_map_parallel_threshold() {
            Zip::from(&mut output)
                .and(input)
                .and(&slopes)
                .par_for_each(p_relu);
        } else {
            Zip::from(&mut output)
                .and(input)
                .and(&slopes)
                .for_each(p_relu);
        }
        Ok(output)
    }
}

impl LayerBase for PReLU {
    fn layer_type(&self) -> &str {
        "PReLU"
    }

    fn param_count(&self) -> ParamCounts {
        ParamCounts::trainable(self.alpha.len())
    }

    fn parameters_mut(&mut self) -> Vec<ParamRef<'_>> {
        let Self { alpha, .. } = self;
        vec![ParamRef::no_decay(
            "alpha",
            alpha
                .as_slice_mut()
                .expect("the slopes are kept in C order"),
        )]
    }

    built_layer_shape_functions!();

    named_weight_layer_functions!(
        trainable "alpha" => alpha,
    );
}

impl UnaryLayer for PReLU {
    /// Allocates 1 slope per position of the input shape, with the batch axis removed
    fn build(&mut self, input: &Shape) -> Result<(), Error> {
        let Some(built) = start_build(&self.built, "PReLU", input)? else {
            return Ok(());
        };
        built.check_min_rank("PReLU", 2)?;
        let (_, tail) = built.split_batch("PReLU")?;
        if let Some(axis) = tail.iter().position(|&extent| extent == 0) {
            return Err(Error::invalid_input(format!(
                "PReLU layer expects every dimension of the input shape to be 1 or more: axis \
                 {} has extent 0",
                axis + 1
            )));
        }
        if let Some(&axis) = self.shared_axes.iter().find(|&&axis| axis > tail.len()) {
            return Err(Error::invalid_parameter(
                "shared_axes",
                format!(
                    "holds axis {axis}, and the input has rank {}",
                    tail.len() + 1
                ),
            ));
        }

        // The slope array reads an extent by position, and the batch axis is not part of it, so
        // a placeholder stands in for the batch axis
        let mut dims = vec![1];
        dims.extend(tail);
        self.input_shape = dims;
        self.built = Some(built);
        self.allocate_alpha();
        Ok(())
    }

    /// Training forward: caches the input, which the backward pass needs for both gradients
    ///
    /// The output alone cannot serve. A slope of 0 maps the whole negative side onto 0, which
    /// erases which elements of the input were negative
    fn forward(&self, input: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        if self.built.is_none() {
            return Err(Error::not_built("PReLU"));
        }
        let output = self.activate(input)?;
        if ctx.is_training() {
            ctx.push_cache("PReLU", input.clone());
        }
        Ok(output)
    }

    /// Splits the upstream gradient between the input and the slopes
    ///
    /// The input gradient passes `g` unchanged where the input was above 0. It scales `g` by
    /// the slope where the input was below 0, and it is 0 at exactly 0. The slope gradient sums
    /// `g * x` over every negative element that the slope covers, which is the batch axis and
    /// every shared axis
    fn backward(&self, grad_output: &Tensor, ctx: &mut Ctx) -> Result<Tensor, Error> {
        let Self {
            shared_axes, alpha, ..
        } = self;

        let input: Tensor = ctx.pop_cache("PReLU")?;
        if grad_output.shape() != input.shape() {
            return Err(Error::shape_mismatch(input.shape(), grad_output.shape()));
        }

        let slopes = alpha
            .broadcast(input.raw_dim())
            .expect("the cached input has a shape the slope array covers");
        let mut grad_input = Tensor::zeros(input.raw_dim());
        // Every negative element contributes `g * x` to the slope that covers it. Collecting
        // the contributions here keeps the reduction below a plain sum over whole axes
        let mut contribution = Tensor::zeros(input.raw_dim());
        let split = |dx: &mut f32, share: &mut f32, &x: &f32, &g: &f32, &a: &f32| {
            if x > 0.0 {
                *dx = g;
            } else if x < 0.0 {
                *dx = g * a;
                *share = g * x;
            }
        };
        if input.len() >= cheap_map_parallel_threshold() {
            Zip::from(&mut grad_input)
                .and(&mut contribution)
                .and(&input)
                .and(grad_output)
                .and(&slopes)
                .par_for_each(split);
        } else {
            Zip::from(&mut grad_input)
                .and(&mut contribution)
                .and(&input)
                .and(grad_output)
                .and(&slopes)
                .for_each(split);
        }

        // The batch axis always collapses. A shared axis collapses too, but it keeps extent 1
        // so that the result still matches the slope array. Putting the axis straight
        // back holds the rank constant, so no later axis index shifts
        let mut reduced = contribution.sum_axis(Axis(0));
        for &axis in shared_axes.iter() {
            let target = Axis(axis - 1);
            reduced = reduced.sum_axis(target).insert_axis(target);
        }

        ctx.add_grad("alpha", reduced.as_standard_layout().to_owned().into_dyn())?;

        Ok(grad_input)
    }

    /// The layer keeps every extent, and it refuses a shape its slope array cannot cover
    ///
    /// The layer changes no extent, so an unbuilt layer answers with the shape it is given.
    /// It still refuses a rank that its `shared_axes` reach past, because that count comes
    /// from the configuration. A built layer holds 1 slope per position of every axis that
    /// `shared_axes` leaves out, and it refuses any other extent on such an axis
    fn compute_output_shape(&self, input: &Shape) -> Result<Shape, Error> {
        if self.built.is_none() {
            input.check_min_rank("PReLU", 2)?;
            if let Some(&axis) = self.shared_axes.iter().find(|&&axis| axis >= input.rank()) {
                return Err(Error::invalid_parameter(
                    "shared_axes",
                    format!("holds axis {axis}, and the input has rank {}", input.rank()),
                ));
            }
            return Ok(input.clone());
        }
        input.check_rank("PReLU", self.input_shape.len())?;
        for (axis, extent) in input.axes().iter().enumerate().skip(1) {
            if self.shared_axes.contains(&axis) {
                continue;
            }
            let wanted = self.input_shape[axis];
            match extent {
                Some(extent) if *extent == wanted => {}
                Some(extent) => {
                    return Err(Error::invalid_input(format!(
                        "PReLU layer holds 1 slope per position of axis {axis}, so that axis \
                         must have extent {wanted}, got {extent}. Add the axis to shared_axes \
                         to accept any extent"
                    )));
                }
                None => {
                    return Err(Error::invalid_input(format!(
                        "PReLU layer holds 1 slope per position of axis {axis}, so that axis \
                         needs a fixed extent. Add the axis to shared_axes to accept any extent"
                    )));
                }
            }
        }
        Ok(input.clone())
    }
}
