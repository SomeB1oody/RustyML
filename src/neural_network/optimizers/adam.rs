//! Adam (Adaptive Moment Estimation) optimizer

use crate::error::Error;
use crate::neural_network::optimizers::kernels;
use crate::neural_network::optimizers::validation::{
    validate_decay_rate, validate_epsilon, validate_learning_rate,
};
use crate::neural_network::traits::{Layer, Optimizer};

/// Adam's per-parameter first/second moment buffers, sized lazily on first use
#[derive(Debug, Clone, Default)]
struct AdamParamState {
    m: Vec<f32>,
    v: Vec<f32>,
}

/// Adam (Adaptive Moment Estimation) optimizer
///
/// Computes adaptive learning rates using estimates of first and second moments of gradients
#[derive(Debug)]
pub struct Adam {
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
}

impl Adam {
    /// Creates a new Adam optimizer with the specified hyperparameters
    ///
    /// Validates hyperparameters and initializes internal timestep tracking
    ///
    /// # Parameters
    ///
    /// - `learning_rate` - Step size for parameter updates
    /// - `beta1` - Decay rate for the first moment estimates (typically 0.9)
    /// - `beta2` - Decay rate for the second moment estimates (typically 0.999)
    /// - `epsilon` - Small constant for numerical stability (typically 1e-8)
    ///
    /// # Returns
    ///
    /// - `Result<Self, Error>` - A new Adam optimizer instance or an error
    ///
    /// # Errors
    ///
    /// - `Error::InvalidParameter` - If any hyperparameter is out of range
    pub fn new(learning_rate: f32, beta1: f32, beta2: f32, epsilon: f32) -> Result<Self, Error> {
        validate_learning_rate(learning_rate)?;
        validate_decay_rate(beta1, "beta1")?;
        validate_decay_rate(beta2, "beta2")?;
        validate_epsilon(epsilon)?;

        Ok(Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            t: 0,
            states: Vec::new(),
            cursor: 0,
        })
    }
}

impl Optimizer for Adam {
    fn step(&mut self) {
        // Advance the timestep once per batch; saturating at i32::MAX keeps the later `t as i32`
        // casts in bias correction from wrapping to a negative exponent
        self.t = self.t.saturating_add(1).min(i32::MAX as u64);
        // Rewind to the first parameter; layers yield parameters in the same order every step
        self.cursor = 0;
    }

    fn update(&mut self, layer: &mut dyn Layer) {
        for pg in layer.parameters() {
            if self.cursor >= self.states.len() {
                self.states.push(AdamParamState {
                    m: vec![0.0; pg.value.len()],
                    v: vec![0.0; pg.value.len()],
                });
            }
            let state = &mut self.states[self.cursor];
            kernels::adam_step(
                pg.value,
                pg.grad,
                &mut state.m,
                &mut state.v,
                self.learning_rate,
                self.beta1,
                self.beta2,
                self.epsilon,
                self.t,
            );
            self.cursor += 1;
        }
    }
}
