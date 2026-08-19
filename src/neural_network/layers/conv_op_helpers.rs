//! The depthwise convolution kernel, shared by the depthwise and separable convolution layers
//!
//! 4 layers share this module: `DepthwiseConv1D` and `DepthwiseConv2D`, plus the first stage of
//! `SeparableConv1D` and `SeparableConv2D`
//!
//! A depthwise convolution never mixes channels. Under the crate's channels-last layout, the
//! channel axis is a pure vector lane. One kernel tap at one output position reads `channels`
//! contiguous input floats, and writes `channels * depth_multiplier` contiguous accumulator
//! floats. These kernels exploit that layout, so they take flat slices instead of per-channel
//! plane views. A channels-first plane would need a strided gather, which this layout avoids
//!
//! The kernels skip out-of-range taps instead of materializing a padded copy, so `Same` padding
//! costs no extra buffer.
//!
//! The geometry below names 2 spatial axes, but the 1D layers use it too. A
//! `[batch, length, channels]` tensor holds the same values in the same row-major order as
//! `[batch, 1, length, channels]`. A `[kernel, channels, depth_multiplier]` weight tensor holds
//! the same order as `[1, kernel, channels, depth_multiplier]`. A 1D layer therefore sets the
//! height fields to 1 and hands over the flat slices of its own rank-3 arrays. No repacking
//! runs, and the module needs no separate 1D loop nest

use crate::parallel_gates::naive_conv_parallel_min_flops;
use rayon::prelude::*;

/// The shape and stride facts a depthwise pass needs, shared by its forward and backward kernels
pub(super) struct DepthwiseGeometry {
    /// Input spatial extent as (height, width)
    pub input: (usize, usize),
    /// Output spatial extent as (height, width)
    pub output: (usize, usize),
    /// Input channels
    pub channels: usize,
    /// Kernels per input channel
    pub depth_multiplier: usize,
    /// Kernel extent as (height, width)
    pub kernel: (usize, usize),
    /// Stride as (height, width)
    pub strides: (usize, usize),
    /// Leading zero-padding as (top, left)
    pub pad_before: (usize, usize),
}

impl DepthwiseGeometry {
    /// Output channel count, `channels * depth_multiplier`
    pub fn out_channels(&self) -> usize {
        self.channels * self.depth_multiplier
    }

    /// Elements per input batch item
    fn input_item(&self) -> usize {
        self.input.0 * self.input.1 * self.channels
    }

    /// Elements per output batch item
    fn output_item(&self) -> usize {
        self.output.0 * self.output.1 * self.out_channels()
    }

    /// Flat offset of the input position a window at `(oh, ow)` reads for tap `(kh, kw)`, or
    /// `None` when that tap falls in the zero padding
    #[inline]
    fn tap_offset(&self, oh: usize, ow: usize, kh: usize, kw: usize) -> Option<usize> {
        let ih = (oh * self.strides.0 + kh).checked_sub(self.pad_before.0)?;
        let iw = (ow * self.strides.1 + kw).checked_sub(self.pad_before.1)?;
        if ih >= self.input.0 || iw >= self.input.1 {
            return None;
        }
        Some((ih * self.input.1 + iw) * self.channels)
    }
}

/// Runs the forward pass of a depthwise convolution over a whole batch
///
/// `src` is the flat `[batch, height, width, channels]` input and `ker` the flat
/// `[kh, kw, channels, depth_multiplier]` kernel. `out` is the flat
/// `[batch, out_height, out_width, channels * depth_multiplier]` output, which this fills.
/// `bias` seeds every output position when the caller has one. A caller whose bias belongs to a
/// later stage passes `None`
///
/// The work splits into 1 task per (batch item, output row). Output rows are disjoint, so this
/// needs no halo and no merge. It keeps every core busy even at `batch == 1`
pub(super) fn depthwise_forward(
    g: &DepthwiseGeometry,
    src: &[f32],
    ker: &[f32],
    bias: Option<&[f32]>,
    out: &mut [f32],
) {
    // `out.len()` is `batch * out_height * out_width * out_channels`, so this is the same
    // estimate as a product written out term by term
    let flops = 2 * out.len() * g.kernel.0 * g.kernel.1;
    let row_len = g.output.1 * g.out_channels();

    if flops >= naive_conv_parallel_min_flops() {
        out.par_chunks_mut(row_len)
            .enumerate()
            .for_each(|(i, row)| {
                depthwise_forward_row(g, src, ker, bias, i / g.output.0, i % g.output.0, row)
            });
    } else {
        for (i, row) in out.chunks_mut(row_len).enumerate() {
            depthwise_forward_row(g, src, ker, bias, i / g.output.0, i % g.output.0, row);
        }
    }
}

/// Fills one output row of a depthwise convolution
///
/// `src` is the whole `[batch, height, width, channels]` input and `ker` the whole
/// `[kh, kw, channels, depth_multiplier]` kernel, both flat and row-major. `out_row` is the
/// `[out_width, channels * depth_multiplier]` row for output row `oh` of batch item `b`.
/// `bias` seeds it when the caller has one. Otherwise the row starts at zero
fn depthwise_forward_row(
    g: &DepthwiseGeometry,
    src: &[f32],
    ker: &[f32],
    bias: Option<&[f32]>,
    b: usize,
    oh: usize,
    out_row: &mut [f32],
) {
    let (kh_size, kw_size) = g.kernel;
    let out_channels = g.out_channels();
    let dm = g.depth_multiplier;
    let in_item = b * g.input_item();

    for ow in 0..g.output.1 {
        let acc = &mut out_row[ow * out_channels..(ow + 1) * out_channels];
        match bias {
            Some(bias) => acc.copy_from_slice(bias),
            None => acc.fill(0.0),
        }
        for kh in 0..kh_size {
            for kw in 0..kw_size {
                let Some(off) = g.tap_offset(oh, ow, kh, kw) else {
                    continue;
                };
                let x = &src[in_item + off..][..g.channels];
                let k = &ker[(kh * kw_size + kw) * out_channels..][..out_channels];
                if dm == 1 {
                    // The general form below is `for m in 0..dm`, whose trip count the compiler
                    // cannot see is 1, so it emits a scalar nested loop. This fast path is
                    // instead a plain element-wise multiply-accumulate over 3 equal-length
                    // contiguous slices. `depth_multiplier == 1` is Keras' default
                    for ((a, &xc), &kc) in acc.iter_mut().zip(x).zip(k) {
                        *a += xc * kc;
                    }
                } else {
                    for (c, &xc) in x.iter().enumerate() {
                        let base = c * dm;
                        for m in 0..dm {
                            acc[base + m] += xc * k[base + m];
                        }
                    }
                }
            }
        }
    }
}

/// The gradients of a depthwise convolution
///
/// The 2 producers below fill the same struct. `depthwise_item_gradients` covers 1 batch item,
/// so its `input` field is 1 item long. [`depthwise_backward`] covers a whole batch, so its
/// `weight` and `bias` fields are already summed over the batch and its `input` field carries
/// every item
pub(super) struct DepthwiseGradients {
    /// Weight gradient, flat `[kh, kw, channels, depth_multiplier]`
    pub weight: Vec<f32>,
    /// Bias gradient, one value per output channel
    pub bias: Vec<f32>,
    /// Input gradient, flat `[height, width, channels]` per batch item
    pub input: Vec<f32>,
}

/// Weight, bias, and input gradients of a depthwise convolution over a whole batch
///
/// `src` is the flat `[batch, height, width, channels]` input, `grad` the flat gradient with
/// respect to this convolution's output, and `ker` the flat kernel. The returned `input` field
/// is the flat gradient with respect to `src`, in the same layout
///
/// The work splits by batch item, which keeps every write private. This sums the weight and bias
/// partials in batch order, so the result does not depend on whether the parallel branch ran.
/// A caller whose bias belongs to a later stage (the depthwise stage of a separable convolution
/// carries its bias on the pointwise side) ignores the `bias` field
pub(super) fn depthwise_backward(
    g: &DepthwiseGeometry,
    src: &[f32],
    grad: &[f32],
    ker: &[f32],
    batch_size: usize,
) -> DepthwiseGradients {
    let flops =
        2 * batch_size * g.out_channels() * g.output.0 * g.output.1 * g.kernel.0 * g.kernel.1;

    let run = |b: usize| depthwise_item_gradients(g, src, grad, ker, b);
    let per_item: Vec<DepthwiseGradients> = if flops >= naive_conv_parallel_min_flops() {
        (0..batch_size).into_par_iter().map(run).collect()
    } else {
        (0..batch_size).map(run).collect()
    };

    let mut weight = vec![0.0f32; g.kernel.0 * g.kernel.1 * g.out_channels()];
    let mut bias = vec![0.0f32; g.out_channels()];
    let mut input = Vec::with_capacity(batch_size * g.input_item());
    for part in per_item {
        for (acc, v) in weight.iter_mut().zip(part.weight) {
            *acc += v;
        }
        for (acc, v) in bias.iter_mut().zip(part.bias) {
            *acc += v;
        }
        input.extend(part.input);
    }

    DepthwiseGradients {
        weight,
        bias,
        input,
    }
}

/// Weight, bias and input gradients for one batch item of a depthwise convolution
fn depthwise_item_gradients(
    g: &DepthwiseGeometry,
    src: &[f32],
    grad: &[f32],
    ker: &[f32],
    b: usize,
) -> DepthwiseGradients {
    let (kh_size, kw_size) = g.kernel;
    let out_channels = g.out_channels();
    let dm = g.depth_multiplier;

    let mut weight = vec![0.0f32; kh_size * kw_size * out_channels];
    let mut bias = vec![0.0f32; out_channels];
    let mut input = vec![0.0f32; g.input_item()];

    let in_item = b * g.input_item();
    let g_item = b * g.output_item();
    for oh in 0..g.output.0 {
        for ow in 0..g.output.1 {
            let gr = &grad[g_item + (oh * g.output.1 + ow) * out_channels..][..out_channels];
            for (j, &gj) in gr.iter().enumerate() {
                bias[j] += gj;
            }
            for kh in 0..kh_size {
                for kw in 0..kw_size {
                    let Some(off) = g.tap_offset(oh, ow, kh, kw) else {
                        continue;
                    };
                    let k_off = (kh * kw_size + kw) * out_channels;
                    if dm == 1 {
                        // Same reason as the forward's fast path. With the multiplier fixed at
                        // 1, every index collapses to the channel, and all 5 slices are
                        // equal-length and contiguous
                        let x = &src[in_item + off..][..g.channels];
                        let wg = &mut weight[k_off..][..g.channels];
                        let kc = &ker[k_off..][..g.channels];
                        let dx = &mut input[off..][..g.channels];
                        for ((((w, d), &xc), &kv), &gj) in
                            wg.iter_mut().zip(dx.iter_mut()).zip(x).zip(kc).zip(gr)
                        {
                            *w += xc * gj;
                            *d += kv * gj;
                        }
                    } else {
                        for c in 0..g.channels {
                            let xc = src[in_item + off + c];
                            let base = c * dm;
                            let mut dxc = 0.0f32;
                            for m in 0..dm {
                                let gj = gr[base + m];
                                weight[k_off + base + m] += xc * gj;
                                dxc += ker[k_off + base + m] * gj;
                            }
                            input[off + c] += dxc;
                        }
                    }
                }
            }
        }
    }

    DepthwiseGradients {
        weight,
        bias,
        input,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tap_offset` skips a tap that reaches past the input, in either direction, rather than
    /// clamping it. This lets `Same` padding work without a padded copy
    #[test]
    fn tap_offset_skips_padding_on_both_edges() {
        let g = DepthwiseGeometry {
            input: (3, 3),
            output: (3, 3),
            channels: 2,
            depth_multiplier: 1,
            kernel: (3, 3),
            strides: (1, 1),
            pad_before: (1, 1),
        };

        // Output (0, 0) with tap (0, 0) sits 1 row and 1 column before the input
        assert_eq!(g.tap_offset(0, 0, 0, 0), None);
        // Tap (1, 1) of the same window is input (0, 0)
        assert_eq!(g.tap_offset(0, 0, 1, 1), Some(0));
        // Output (2, 2) with tap (2, 2) runs 1 past the trailing edge
        assert_eq!(g.tap_offset(2, 2, 2, 2), None);
        // Its tap (1, 1) is input (2, 2), the last position: (2 * 3 + 2) * 2 channels
        assert_eq!(g.tap_offset(2, 2, 1, 1), Some(16));
    }
}
