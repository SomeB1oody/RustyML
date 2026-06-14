//! Shared state and update machinery for the Adam-family optimizers (`Adam`, `AdamW`)
//!
//! Both optimizers run identical Adam moment math and bias correction; they differ only in how a
//! non-zero `weight_decay` enters the update — classic [`Adam`](super::adam::Adam) folds an L2
//! penalty into the gradient (coupled), while [`AdamW`](super::adam_w::AdamW) shrinks the
//! parameter directly (decoupled). That single difference is the `decoupled` flag; everything else
//! (hyperparameter validation, clip-norm, the bias-correction timestep, and the lazily-sized
//! per-parameter moment buffers) lives here once

use crate::error::Error;
use crate::neural_network::optimizers::kernels;
use crate::neural_network::optimizers::validation::{
    validate_clip_norm, validate_decay_rate, validate_epsilon, validate_learning_rate,
    validate_non_negative_finite,
};
use crate::neural_network::traits::Layer;

/// Adam's per-parameter first/second moment buffers, sized lazily on first use
#[derive(Debug, Clone, Default)]
struct AdamParamState {
    m: Vec<f32>,
    v: Vec<f32>,
}

/// Shared Adam-family optimizer state
///
/// `decoupled` selects the weight-decay flavor: `true` is AdamW (decoupled decay, applied directly
/// to the parameter), `false` is classic Adam (coupled L2 decay, folded into the gradient). With
/// `weight_decay == 0.0` the flag has no effect and both reduce to plain Adam
#[derive(Debug)]
pub(super) struct AdamCore {
    /// Learning rate controlling the size of parameter updates
    learning_rate: f32,
    /// Exponential decay rate for the first moment estimates
    beta1: f32,
    /// Exponential decay rate for the second moment estimates
    beta2: f32,
    /// Small constant added for numerical stability
    epsilon: f32,
    /// Current timestep, incremented with each update
    t: u64,
    /// Per-parameter moment buffers, indexed by the order layers yield parameters each step
    states: Vec<AdamParamState>,
    /// Position within `states` for the parameter currently being updated; reset each `step`
    cursor: usize,
    /// Optional clip-by-global-norm threshold; `None` disables gradient clipping
    clip_norm: Option<f32>,
    /// Weight decay coefficient; `0.0` disables it
    weight_decay: f32,
    /// `true` for AdamW (decoupled decay), `false` for classic Adam (coupled L2 decay)
    decoupled: bool,
}

impl AdamCore {
    /// Validates the hyperparameters and builds the core in the given decay mode
    pub(super) fn new(
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
        decoupled: bool,
    ) -> Result<Self, Error> {
        validate_learning_rate(learning_rate)?;
        validate_decay_rate(beta1, "beta1")?;
        validate_decay_rate(beta2, "beta2")?;
        validate_epsilon(epsilon)?;
        validate_non_negative_finite(weight_decay, "weight_decay")?;

        Ok(Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t: 0,
            states: Vec::new(),
            cursor: 0,
            clip_norm: None,
            weight_decay,
            decoupled,
        })
    }

    /// Enables clip-by-global-norm gradient clipping (consuming builder)
    pub(super) fn with_clip_norm(mut self, clip_norm: f32) -> Result<Self, Error> {
        validate_clip_norm(Some(clip_norm))?;
        self.clip_norm = Some(clip_norm);
        Ok(self)
    }

    /// The configured clip-by-global-norm threshold, or `None` when disabled
    pub(super) fn clip_norm(&self) -> Option<f32> {
        self.clip_norm
    }

    /// Retunes the step size, preserving all accumulated moment state
    pub(super) fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }

    /// Advances the bias-correction timestep once per batch and rewinds the parameter cursor
    pub(super) fn step(&mut self) {
        // Advance the timestep once per batch
        self.t = self.t.saturating_add(1).min(i32::MAX as u64);
        // Rewind to the first parameter
        self.cursor = 0;
    }

    /// Updates one layer's parameters, applying weight decay per the `decoupled` mode
    pub(super) fn update(&mut self, layer: &mut dyn Layer, grad_scale: f32) {
        for pg in layer.parameters() {
            if self.cursor >= self.states.len() {
                self.states.push(AdamParamState {
                    m: vec![0.0; pg.value.len()],
                    v: vec![0.0; pg.value.len()],
                });
            } else if self.states[self.cursor].m.len() != pg.value.len() {
                // Reset the moment buffers to match
                self.states[self.cursor] = AdamParamState {
                    m: vec![0.0; pg.value.len()],
                    v: vec![0.0; pg.value.len()],
                };
            }
            let state = &mut self.states[self.cursor];
            let grad = kernels::scaled_grad(pg.grad, grad_scale);

            // Weight decay applies to weight tensors only; biases and normalization gamma/beta
            // carry `decays = false` and are never decayed
            if pg.decays && self.weight_decay != 0.0 {
                if self.decoupled {
                    // AdamW: decoupled decay shrinks the parameter before a plain Adam step
                    kernels::apply_weight_decay(pg.value, self.learning_rate, self.weight_decay);
                    kernels::adam_step(
                        pg.value,
                        &grad,
                        &mut state.m,
                        &mut state.v,
                        self.learning_rate,
                        self.beta1,
                        self.beta2,
                        self.epsilon,
                        self.t,
                    );
                } else {
                    // Adam: coupled L2 decay folds weight_decay * param into the gradient
                    let l2 = kernels::l2_regularized_grad(&grad, pg.value, self.weight_decay);
                    kernels::adam_step(
                        pg.value,
                        &l2,
                        &mut state.m,
                        &mut state.v,
                        self.learning_rate,
                        self.beta1,
                        self.beta2,
                        self.epsilon,
                        self.t,
                    );
                }
            } else {
                kernels::adam_step(
                    pg.value,
                    &grad,
                    &mut state.m,
                    &mut state.v,
                    self.learning_rate,
                    self.beta1,
                    self.beta2,
                    self.epsilon,
                    self.t,
                );
            }
            self.cursor += 1;
        }
    }
}
