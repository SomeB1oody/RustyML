//! Shared state and update machinery for the Adam-family optimizers (`Adam`, `AdamW`)
//!
//! Both optimizers run identical Adam moment math and bias correction. They differ only in how a
//! non-zero `weight_decay` enters the update. Classic [`Adam`](super::adam::Adam) folds an L2
//! penalty into the gradient (coupled), while [`AdamW`](super::adam_w::AdamW) shrinks the
//! parameter directly (decoupled). That single difference is the `decoupled` flag. Everything else
//! (hyperparameter validation, clip-by-global-norm clipping, the bias-correction timestep, and the
//! lazily-sized per-parameter moment buffers) lives here once

use crate::error::Error;
use crate::neural_network::ctx::Grads;
use crate::neural_network::optimizers::kernels;
use crate::neural_network::optimizers::validation::{
    validate_decay_rate, validate_epsilon, validate_global_clipnorm, validate_learning_rate,
    validate_non_negative_finite,
};
use crate::neural_network::traits::{LayerBase, ParamId};
use std::collections::HashMap;

/// Adam's per-parameter first/second moment buffers, sized lazily on first use
#[derive(Debug, Clone, Default)]
struct AdamParamState {
    /// First-moment (mean) estimate
    m: Vec<f32>,
    /// Second-moment (uncentered variance) estimate
    v: Vec<f32>,
}

/// Shared Adam-family optimizer state
///
/// `decoupled` selects the weight-decay flavor. `true` is AdamW (decoupled decay, applied
/// directly to the parameter). `false` is classic Adam (coupled L2 decay, folded into the
/// gradient). With `weight_decay == 0.0` the flag has no effect and both reduce to plain Adam
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
    /// Per-parameter moment buffers, keyed by the address of the parameter. A buffer therefore
    /// follows its own tensor, whatever else the model does
    states: HashMap<ParamId, AdamParamState>,
    /// Optional clip-by-global-norm threshold. `None` disables gradient clipping
    global_clipnorm: Option<f32>,
    /// Weight decay coefficient. `0.0` disables it
    weight_decay: f32,
    /// `true` for AdamW (decoupled decay), `false` for classic Adam (coupled L2 decay)
    decoupled: bool,
}

impl AdamCore {
    /// Validates the hyperparameters and builds the core in the given decay mode
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - step size for parameter updates
    /// - `beta1` - decay rate for the first moment estimates
    /// - `beta2` - decay rate for the second moment estimates
    /// - `epsilon` - denominator stabilizer for numerical stability
    /// - `weight_decay` - weight-decay coefficient. `0.0` disables it
    /// - `decoupled` - `true` builds the core in AdamW mode (decoupled decay), `false` builds it
    ///   in classic Adam mode (coupled L2 decay)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - the built core, or an error if a hyperparameter is out of range
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
            states: HashMap::new(),
            global_clipnorm: None,
            weight_decay,
            decoupled,
        })
    }

    /// Enables clip-by-global-norm gradient clipping (consuming builder)
    ///
    /// # Parameters
    ///
    /// - `global_clipnorm` - clip-by-global-norm threshold. Must be positive and finite
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - the updated core, or an error if `global_clipnorm` is not
    ///   positive and finite
    pub(super) fn with_global_clipnorm(mut self, global_clipnorm: f32) -> Result<Self, Error> {
        validate_global_clipnorm(Some(global_clipnorm))?;
        self.global_clipnorm = Some(global_clipnorm);
        Ok(self)
    }

    /// The configured clip-by-global-norm threshold, or `None` when disabled
    pub(super) fn global_clipnorm(&self) -> Option<f32> {
        self.global_clipnorm
    }

    /// The current step size
    pub(super) fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    /// Retunes the step size, preserving all accumulated moment state
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - the new learning rate to use for subsequent updates
    pub(super) fn set_learning_rate(&mut self, learning_rate: f32) {
        self.learning_rate = learning_rate;
    }

    /// Advances the bias-correction timestep once per batch
    pub(super) fn step(&mut self) {
        // Clamp at i32::MAX so the bias-correction power (`t as i32`) stays valid
        self.t = self.t.saturating_add(1).min(i32::MAX as u64);
    }

    /// Updates a layer's parameters, applying weight decay per the `decoupled` mode
    ///
    /// # Parameters
    ///
    /// - `scope` - position of this layer in the model, counted from the input. It is the layer
    ///   half of the parameter address
    /// - `layer` - the layer whose parameters should be updated
    /// - `grads` - every gradient the backward pass produced
    /// - `grad_scale` - uniform factor applied to every gradient before the update, to implement
    ///   clip-by-global-norm. Pass `1.0` for an unscaled update
    pub(super) fn update(
        &mut self,
        scope: usize,
        layer: &mut dyn LayerBase,
        grads: &Grads,
        grad_scale: f32,
    ) {
        for pg in layer.parameters_mut() {
            let Some(grad) = grads.get(ParamId::new(scope, pg.name)) else {
                continue;
            };
            let grad = grad
                .as_slice()
                .expect("a stored gradient is in the standard memory order");
            debug_assert_eq!(grad.len(), pg.value.len());
            let state = self.states.entry(ParamId::new(scope, pg.name)).or_default();
            if state.m.len() != pg.value.len() {
                // The tensor was resized under its own name: start the moment buffers again
                *state = AdamParamState {
                    m: vec![0.0; pg.value.len()],
                    v: vec![0.0; pg.value.len()],
                };
            }
            let grad = kernels::scaled_grad(grad, grad_scale);

            // Weight decay applies to weight tensors only. Biases and normalization gamma/beta
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
        }
    }
}
