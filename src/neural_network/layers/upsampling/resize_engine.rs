//! Resize kernels shared by every upsampling layer
//!
//! An upsampling layer enlarges 1 spatial axis at a time. Each axis pass is a gather. Every
//! position of the enlarged axis reads a short run of neighboring positions of the source axis
//! and adds them up under fixed weights. A [`Band`] holds those weights for 1 axis
//!
//! The pass is linear, so the backward pass is the same gather under the transposed band. That
//! keeps 1 kernel for both directions, and it keeps the backward pass free of any scatter
//!
//! Every function here indexes `factors` by spatial axis. Entry `i` of `factors` describes axis
//! `i + 1` of the tensor. The batch axis and the channel axis have no entry, because an
//! upsampling layer never changes them

use crate::error::Error;
use crate::neural_network::Tensor;
use crate::neural_network::layers::upsampling::Interpolation;
use ndarray::IxDyn;
use rayon::iter::{IndexedParallelIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;

tunable_gate! {
    /// Total element ops (`destination elements * taps`) at or above which 1 axis pass runs in
    /// parallel
    ///
    /// The gate counts element ops rather than elements, because the taps per output position
    /// vary with the mode and the direction. A forward repeat pass reads 1 tap, and a forward
    /// interpolated pass reads up to 11, the `Lanczos5` count. A backward pass reads more taps
    /// than its forward pass, and the count grows with the factor. An element count alone would
    /// put a wide pass and a narrow pass of the same output size on the same side of the gate.
    /// One of them can do far more work
    ///
    /// The default is the point at which every mode at least breaks even against its own serial
    /// path. Below it the repeat mode loses to the rayon task overhead, because it moves memory
    /// and computes almost nothing. Above it every mode gains, and the wide kernels gain most
    ///
    /// Overridable through [`crate::tuning`]
    pub(crate) UPSAMPLE_PARALLEL_MIN_OPS
        => upsample_parallel_min_ops / set_upsample_parallel_min_ops = 2_000_000
}

/// Positions the widest kernel reads per axis, which is the radius 5 of `Lanczos5` on both sides
const MAX_TAPS: usize = 11;

/// Smallest weight sum the pass still normalizes
///
/// The pass divides a position's weights by their sum when the sum is at least this bound.
/// Below the bound, the position keeps a weight of 0 for every tap. The bound is 1000 times the
/// `f32` epsilon
const MIN_WEIGHT_SUM: f64 = 1000.0 * f32::EPSILON as f64;

/// A resize kernel, read at a distance that is never negative
type Kernel = fn(f64) -> f64;

/// A weighted gather along 1 axis
///
/// Each destination position reads `taps` source positions in a row, the first of them named by
/// `starts`. In a forward band, the weights of 1 destination position add up to 1. A backward
/// band regroups those same weights by source position, so its rows need not add up to 1
struct Band {
    /// Extent of the axis after the pass
    out_len: usize,
    /// Source positions each destination position reads
    taps: usize,
    /// First source position of each destination position, 1 entry per destination position
    starts: Vec<usize>,
    /// `taps` weights per destination position, or `None` when every weight is 1
    weights: Option<Vec<f32>>,
}

/// Triangle kernel of radius 1, which gives linear interpolation
fn triangle(x: f64) -> f64 {
    (1.0 - x).max(0.0)
}

/// Keys cubic kernel of radius 2, with `a = -0.5`
///
/// R. G. Keys, "Cubic convolution interpolation for digital image processing", IEEE Transactions
/// on Acoustics, Speech, and Signal Processing, 29(6):1153-1160, 1981
fn keys_cubic(x: f64) -> f64 {
    if x < 1.0 {
        ((1.5 * x - 2.5) * x) * x + 1.0
    } else if x < 2.0 {
        ((-0.5 * x + 2.5) * x - 4.0) * x + 2.0
    } else {
        0.0
    }
}

/// Lanczos kernel of the given radius, which is `sinc(x) * sinc(x / radius)`
///
/// The function returns 1 for a distance at or below `1e-3`, because the quotient loses its
/// precision there
fn lanczos(radius: f64, x: f64) -> f64 {
    if x <= 1e-3 {
        return 1.0;
    }
    if x > radius {
        return 0.0;
    }
    let numerator =
        radius * (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * x / radius).sin();
    numerator / (std::f64::consts::PI * std::f64::consts::PI * x * x)
}

/// The kernel radius and the kernel of an interpolated mode, or `None` for the repeat mode
fn resample_kernel(interpolation: Interpolation) -> Option<(usize, Kernel)> {
    match interpolation {
        Interpolation::Nearest => None,
        Interpolation::Bilinear => Some((1, triangle)),
        Interpolation::Bicubic => Some((2, keys_cubic)),
        Interpolation::Lanczos3 => Some((3, |x| lanczos(3.0, x))),
        Interpolation::Lanczos5 => Some((5, |x| lanczos(5.0, x))),
    }
}

/// Builds the forward weights of 1 interpolated axis
///
/// Returns `(taps, starts, weights)`. Output position `j` sits at the source coordinate
/// `(j + 0.5) / factor - 0.5`. That places the center of an output position at the center of
/// the source region it covers. The pass then reads the kernel at the distance from that
/// coordinate to each source position
///
/// A position near an edge has part of its kernel outside the input. The pass drops those
/// weights and divides the remaining weights by their own sum. That is what keeps an edge from
/// fading toward 0
fn resample_weights(
    in_len: usize,
    factor: usize,
    radius: usize,
    kernel: Kernel,
) -> (usize, Vec<usize>, Vec<f32>) {
    let out_len = in_len * factor;
    let taps = (2 * radius + 1).min(in_len);
    let inv_scale = 1.0 / factor as f64;

    let mut starts = Vec::with_capacity(out_len);
    let mut weights = vec![0.0f32; out_len * taps];
    for (position, row) in weights.chunks_exact_mut(taps).enumerate() {
        let center = (position as f64 + 0.5) * inv_scale - 0.5;
        // The kernel reaches `radius` on each side. Shifting the window to fit the input drops
        // only positions the kernel already weighs at 0
        let lowest = (center - radius as f64).ceil().max(0.0) as usize;
        let start = lowest.min(in_len - taps);

        let mut raw = [0.0f64; MAX_TAPS];
        let mut total = 0.0;
        for (tap, value) in raw[..taps].iter_mut().enumerate() {
            *value = kernel((center - (start + tap) as f64).abs());
            total += *value;
        }
        if total.abs() > MIN_WEIGHT_SUM {
            for (weight, value) in row.iter_mut().zip(&raw[..taps]) {
                *weight = (value / total) as f32;
            }
        }
        starts.push(start);
    }

    (taps, starts, weights)
}

/// Transposes a forward band, which gives the band of the backward pass
///
/// `starts` never decreases, because the source coordinate grows with the output position. The
/// output positions that read source position `i` therefore form 1 run. 2 cursors that only
/// move forward find that run for every `i` in 1 walk
fn transpose_weights(src_len: usize, taps: usize, starts: &[usize], weights: &[f32]) -> Band {
    let out_len = starts.len();
    let mut firsts = vec![0usize; src_len];
    let mut counts = vec![0usize; src_len];
    let (mut low, mut high) = (0usize, 0usize);
    for (i, (first, count)) in firsts.iter_mut().zip(counts.iter_mut()).enumerate() {
        while low < out_len && starts[low] + taps <= i {
            low += 1;
        }
        while high < out_len && starts[high] <= i {
            high += 1;
        }
        *first = low;
        *count = high - low;
    }

    let back_taps = counts.iter().copied().max().unwrap_or(1).max(1);
    let mut back_starts = Vec::with_capacity(src_len);
    let mut back_weights = vec![0.0f32; src_len * back_taps];
    let rows = back_weights.chunks_exact_mut(back_taps);
    for ((i, row), (&first, &count)) in rows.enumerate().zip(firsts.iter().zip(&counts)) {
        // The run must sit inside the axis to be stored at a fixed stride. It is never longer
        // than `back_taps`, so pulling it back from the end still covers all of it
        let start = first.min(out_len - back_taps);
        for position in first..first + count {
            row[position - start] = weights[position * taps + (i - starts[position])];
        }
        back_starts.push(start);
    }

    Band {
        out_len: src_len,
        taps: back_taps,
        starts: back_starts,
        weights: Some(back_weights),
    }
}

/// Builds the band of 1 spatial axis, in the forward or in the backward direction
///
/// Returns `None` for a factor of 1, which leaves the axis alone
fn axis_band(
    in_len: usize,
    factor: usize,
    interpolation: Interpolation,
    backward: bool,
) -> Option<Band> {
    if factor == 1 {
        return None;
    }
    let out_len = in_len * factor;

    let Some((radius, kernel)) = resample_kernel(interpolation) else {
        // The repeat mode copies 1 source position into a run of `factor` positions, so its
        // backward pass adds that run back up. Neither direction needs a weight
        return Some(if backward {
            Band {
                out_len: in_len,
                taps: factor,
                starts: (0..in_len).map(|i| i * factor).collect(),
                weights: None,
            }
        } else {
            Band {
                out_len,
                taps: 1,
                starts: (0..out_len).map(|j| j / factor).collect(),
                weights: None,
            }
        });
    };

    let (taps, starts, weights) = resample_weights(in_len, factor, radius, kernel);
    Some(if backward {
        transpose_weights(in_len, taps, &starts, &weights)
    } else {
        Band {
            out_len,
            taps,
            starts,
            weights: Some(weights),
        }
    })
}

/// Destination elements a task takes at a time, before the per-task setup is paid again
///
/// A task walks its own rows, so it determines the axis position of its first row only. A run
/// this size keeps that setup under 1 percent of the work, for any `inner` value the layers see
const TASK_ELEMENTS: usize = 16_384;

/// Runs 1 band over a C-order buffer seen as `[outer, src_len, inner]`
///
/// Each destination position owns 1 run of `inner` elements and reads only from the source, so
/// the tasks write no shared element. The pass adds the taps in a fixed order, so the result
/// matches between the serial path and the parallel path
fn apply_band(src: &[f32], src_len: usize, inner: usize, band: &Band, dst: &mut [f32]) {
    let rows_per_task = (TASK_ELEMENTS / inner).max(1);
    let task = |(index, dst_task): (usize, &mut [f32])| {
        // Every task but the last is full, so the first row of a task is exact. Counting the
        // rest by hand keeps the walk free of the 2 divisions a flat row index would need
        let first_row = index * rows_per_task;
        let mut lane = first_row / band.out_len;
        let mut position = first_row % band.out_len;

        for dst_row in dst_task.chunks_mut(inner) {
            let base = lane * src_len * inner + band.starts[position] * inner;
            match &band.weights {
                None => {
                    dst_row.copy_from_slice(&src[base..base + inner]);
                    for tap in 1..band.taps {
                        let from = base + tap * inner;
                        for (d, &s) in dst_row.iter_mut().zip(&src[from..from + inner]) {
                            *d += s;
                        }
                    }
                }
                Some(weights) => {
                    let row_weights = &weights[position * band.taps..][..band.taps];
                    for (d, &s) in dst_row.iter_mut().zip(&src[base..base + inner]) {
                        *d = row_weights[0] * s;
                    }
                    for (tap, &weight) in row_weights.iter().enumerate().skip(1) {
                        let from = base + tap * inner;
                        for (d, &s) in dst_row.iter_mut().zip(&src[from..from + inner]) {
                            *d += weight * s;
                        }
                    }
                }
            }

            position += 1;
            if position == band.out_len {
                position = 0;
                lane += 1;
            }
        }
    };

    // 1 multiply-add per tap per destination element is the work metric of the pass
    let chunk = rows_per_task * inner;
    if dst.len() * band.taps >= upsample_parallel_min_ops() {
        dst.par_chunks_mut(chunk).enumerate().for_each(task);
    } else {
        dst.chunks_mut(chunk).enumerate().for_each(task);
    }
}

/// Applies `bands[i]` along axis `i + 1`, and returns the result in C order
fn run_bands(source: &Tensor, bands: &[Option<Band>]) -> Tensor {
    let mut shape = source.shape().to_vec();
    let mut current = source.as_standard_layout().into_owned();

    for (spatial, band) in bands.iter().enumerate() {
        let Some(band) = band else { continue };
        let axis = spatial + 1;
        let inner: usize = shape[axis + 1..].iter().product();
        let outer: usize = shape[..axis].iter().product();

        let mut next = vec![0.0f32; outer * band.out_len * inner];
        apply_band(
            current.as_slice().expect("the buffer is kept in C order"),
            shape[axis],
            inner,
            band,
            &mut next,
        );
        shape[axis] = band.out_len;
        current = Tensor::from_shape_vec(IxDyn(&shape), next).expect("the shape matches the data");
    }

    current
}

/// Shape an upsampled output takes, given the shape that enters the layer
fn upsampled_shape(input_shape: &[usize], factors: &[usize]) -> Vec<usize> {
    let mut shape = input_shape.to_vec();
    for (spatial, &factor) in factors.iter().enumerate() {
        shape[spatial + 1] *= factor;
    }
    shape
}

/// Runs the forward pass of an upsampling layer
///
/// # Parameters
///
/// - `input` - Tensor entering the layer
/// - `factors` - Factor each spatial axis grows by
/// - `interpolation` - How the layer fills the new positions
/// - `rank` - Rank the layer accepts, batch and channel axes included
/// - `layer` - Layer name, used in error messages
///
/// # Returns
///
/// - `Result<Tensor, Error>` - The enlarged tensor, in C order
///
/// # Errors
///
/// - `Error::InvalidInput` - If the input rank is not `rank`, or if the output element count
///   goes past `usize`
/// - `Error::EmptyInput` - If any axis of the input has an extent of 0
pub(super) fn upsample_forward(
    input: &Tensor,
    factors: &[usize],
    interpolation: Interpolation,
    rank: usize,
    layer: &'static str,
) -> Result<Tensor, Error> {
    if input.ndim() != rank {
        return Err(Error::invalid_input(format!(
            "{} layer expects a {}D input, got a {}D tensor",
            layer,
            rank,
            input.ndim()
        )));
    }
    if input.is_empty() {
        return Err(Error::empty_input("input tensor"));
    }

    // A factor is only bounded below at construction, so a large 1 can take the output past
    // what an index can hold. Reporting that is better than the wrap or the panic a plain
    // multiply would give
    let mut elements = input.len();
    for (spatial, &factor) in factors.iter().enumerate() {
        elements = elements.checked_mul(factor).ok_or_else(|| {
            Error::invalid_input(format!(
                "{} layer grows axis {} of a {:?} input by {}, and the output does not fit in memory",
                layer,
                spatial + 1,
                input.shape(),
                factor
            ))
        })?;
    }

    let bands: Vec<Option<Band>> = factors
        .iter()
        .enumerate()
        .map(|(spatial, &factor)| {
            axis_band(input.shape()[spatial + 1], factor, interpolation, false)
        })
        .collect();
    Ok(run_bands(input, &bands))
}

/// Runs the backward pass of an upsampling layer
///
/// Every new position is a weighted sum of source positions. Each source position collects the
/// gradient of every position it fed, under the same weights
///
/// # Parameters
///
/// - `grad_output` - Gradient from the next layer
/// - `input_shape` - Shape of the most recent forward input, or `None` if none has run
/// - `factors` - Factor each spatial axis grew by
/// - `interpolation` - How the forward pass filled the new positions
/// - `layer` - Layer name, used in error messages
///
/// # Returns
///
/// - `Result<Tensor, Error>` - The gradient for the previous layer, in C order
///
/// # Errors
///
/// - `Error::NeuralNetwork(NnError::ForwardPassNotRun)` - If `input_shape` is `None`
/// - `Error::ShapeMismatch` - If `grad_output` does not have the upsampled output shape
pub(super) fn upsample_backward(
    grad_output: &Tensor,
    input_shape: Option<&[usize]>,
    factors: &[usize],
    interpolation: Interpolation,
    layer: &'static str,
) -> Result<Tensor, Error> {
    let Some(input_shape) = input_shape else {
        return Err(Error::forward_pass_not_run(layer));
    };

    let expected = upsampled_shape(input_shape, factors);
    if grad_output.shape() != expected.as_slice() {
        return Err(Error::shape_mismatch(expected, grad_output.shape()));
    }

    let bands: Vec<Option<Band>> = factors
        .iter()
        .enumerate()
        .map(|(spatial, &factor)| axis_band(input_shape[spatial + 1], factor, interpolation, true))
        .collect();
    Ok(run_bands(grad_output, &bands))
}

/// Formats the output shape of an upsampling layer for `summary()`
///
/// An input shape of `None` means no forward pass has run, so the layer cannot know its output
/// shape yet
pub(super) fn upsample_summary(input_shape: Option<&[usize]>, factors: &[usize]) -> String {
    match input_shape {
        Some(shape) => {
            let axes: Vec<String> = upsampled_shape(shape, factors)[1..]
                .iter()
                .map(|extent| extent.to_string())
                .collect();
            format!("(None, {})", axes.join(", "))
        }
        None => "Unknown".to_string(),
    }
}

/// Checks that every upsampling factor is at least 1
///
/// # Errors
///
/// - `Error::InvalidParameter` - If any factor is 0
pub(super) fn validate_factors(factors: &[usize]) -> Result<(), Error> {
    if factors.contains(&0) {
        return Err(Error::invalid_parameter(
            "size",
            "holds a factor of 0, and every factor must be at least 1",
        ));
    }
    Ok(())
}
