//! The contract that every layer type keeps, checked once over the whole roster
//!
//! Each test here walks [`visit_every_layer`] and holds every layer type to 1 rule. 2 rules
//! come from shared code. `Ctx::pop_cache` refuses a backward pass that has no cache, and the
//! checkpoint reads the arrays that `weights` and `weights_mut` give. The 3rd rule is that a
//! layer parks no cache in an inference context. Each layer type keeps this rule in its own
//! code. The test file of 1 layer type therefore tests only the behavior of that type.
//!
//! Each test counts the layer types it covers. A roster case that a test skips without a
//! reason therefore fails that test.

use crate::roster::{Case, LAYER_TYPE_COUNT, LayerVisitor, visit_every_layer};
use rustyml::error::Error;
use rustyml::neural_network::error::NnError;
use rustyml::neural_network::sequential::SequentialBuilder;
use rustyml::neural_network::traits::Layer;
use rustyml::neural_network::{Ctx, Tensor};
use std::collections::BTreeSet;

/// The layer types whose forward pass reads the training flag
///
/// Each of these layers draws a random value, or reads batch statistics, in training mode
/// alone. Its own test file checks the inference behavior
const MODE_DEPENDENT: [&str; 7] = [
    "Dropout",
    "SpatialDropout1D",
    "SpatialDropout2D",
    "SpatialDropout3D",
    "GaussianDropout",
    "GaussianNoise",
    "BatchNormalization",
];

/// The layer types whose backward pass reads no cache
///
/// The gradient of each of these layers is a function of the output gradient alone. A
/// backward pass with no forward pass before it is therefore correct for these layers
const CACHE_FREE: [&str; 3] = ["Rescaling", "Reverse", "GaussianNoise"];

/// The layer type without its configuration, such as `Reverse` for `Reverse(1)`
fn type_of(name: &str) -> &str {
    name.split('(').next().unwrap_or(name)
}

/// A built layer of the case, or a panic that names the layer type
fn built<L: Layer>(case: &Case, make: &dyn Fn() -> L) -> L {
    let mut layer = make();
    layer.build_many(&case.inputs).unwrap_or_else(|error| {
        panic!(
            "{} refused to build for {:?}: {error}",
            layer.layer_type(),
            case.inputs
        )
    });
    layer
}

/// Every layer type refuses a backward pass that no forward pass came before
///
/// The refusal names the layer type, so a model error points at the layer at fault. A layer
/// type in [`CACHE_FREE`] gives the same gradient with no forward pass as after 1. The test
/// then runs a training forward pass and a backward pass. The backward pass takes back every
/// cache that the forward pass parked, so the context holds no cache after it
#[test]
fn every_layer_refuses_a_backward_pass_before_its_forward_pass() {
    struct BackwardNeedsForward(BTreeSet<String>);
    impl LayerVisitor for BackwardNeedsForward {
        fn visit<L: Layer + 'static>(&mut self, case: Case, make: &dyn Fn() -> L) {
            let layer = built(&case, make);
            let name = layer.layer_type().to_string();
            let grad = case.output_grad();

            let unprepared = layer.backward_many(&grad, &mut Ctx::training());
            let cache_free = CACHE_FREE.contains(&type_of(&name));
            match (&unprepared, cache_free) {
                (Ok(_), true) => {}
                (Err(Error::NeuralNetwork(NnError::ForwardPassNotRun(reported))), false) => {
                    assert_eq!(*reported, name, "{name} names another layer type")
                }
                (other, _) => panic!("{name}: expected ForwardPassNotRun, got {other:?}"),
            }

            let sample = case.sample();
            let inputs: Vec<&Tensor> = sample.iter().collect();
            let mut ctx = Ctx::training();
            layer
                .forward_many(&inputs, &mut ctx)
                .unwrap_or_else(|error| panic!("{name} refused its forward pass: {error}"));
            let input_grads = layer
                .backward_many(&grad, &mut ctx)
                .unwrap_or_else(|error| panic!("{name} refused its backward pass: {error}"));
            assert_eq!(input_grads.len(), inputs.len(), "{name}");
            for (input_grad, input) in input_grads.iter().zip(&inputs) {
                assert_eq!(input_grad.shape(), input.shape(), "{name}");
            }
            assert_eq!(
                ctx.pending_caches(),
                0,
                "{name} left a cache in the context"
            );
            if let Ok(unprepared) = unprepared {
                assert_eq!(unprepared, input_grads, "{name}");
            }

            self.0.insert(name);
        }
    }

    let mut visitor = BackwardNeedsForward(BTreeSet::new());
    visit_every_layer(&mut visitor);
    assert_eq!(visitor.0.len(), LAYER_TYPE_COUNT);
}

/// Every layer type parks nothing in inference mode, and a mode-free layer infers what it
/// trains
///
/// An inference context gets no backward pass, so a cache parked there stays until the
/// context drops. For each layer type outside [`MODE_DEPENDENT`], the output of the 2 modes
/// is equal bit for bit
#[test]
fn every_layer_infers_without_a_cache() {
    struct InferenceMatchesTraining(BTreeSet<String>);
    impl LayerVisitor for InferenceMatchesTraining {
        fn visit<L: Layer + 'static>(&mut self, case: Case, make: &dyn Fn() -> L) {
            let layer = built(&case, make);
            let name = layer.layer_type().to_string();
            let sample = case.sample();
            let inputs: Vec<&Tensor> = sample.iter().collect();

            let mut inference = Ctx::inference();
            let inferred = layer
                .forward_many(&inputs, &mut inference)
                .unwrap_or_else(|error| panic!("{name} refused inference: {error}"));
            assert_eq!(
                inference.pending_caches(),
                0,
                "{name} parked a cache in inference mode"
            );
            assert_eq!(
                inference.pending_states(),
                0,
                "{name} wrote state in inference mode"
            );

            if !MODE_DEPENDENT.contains(&type_of(&name)) {
                let trained = layer
                    .forward_many(&inputs, &mut Ctx::training())
                    .unwrap_or_else(|error| panic!("{name} refused training: {error}"));
                assert_eq!(inferred, trained, "{name} gives 2 outputs for 1 input");
            }

            self.0.insert(name);
        }
    }

    let mut visitor = InferenceMatchesTraining(BTreeSet::new());
    visit_every_layer(&mut visitor);
    assert_eq!(visitor.0.len(), LAYER_TYPE_COUNT);
}

/// Every single-input layer type keeps each array through a save and a load
///
/// The test writes a distinct value into each array of a built layer through `weights_mut`.
/// A model saves the layer, and a new model of the same layer type loads the file. Each array
/// and the prediction of the 2 models must then be equal bit for bit. A layer whose `weights`
/// and `weights_mut` disagree on a name, an order, or an array fails here
///
/// A merge layer takes 2 inputs, so a sequential model cannot hold it. A merge layer holds no
/// array, so the checkpoint has nothing of it to keep
#[test]
fn every_layer_round_trips_through_a_checkpoint() {
    struct RoundTrip {
        covered: BTreeSet<String>,
        merge: usize,
    }
    impl LayerVisitor for RoundTrip {
        fn visit<L: Layer + 'static>(&mut self, case: Case, make: &dyn Fn() -> L) {
            let mut layer = built(&case, make);
            let name = layer.layer_type().to_string();
            if case.inputs.len() != 1 {
                assert!(layer.weights().is_empty(), "{name} holds an array");
                self.merge += 1;
                return;
            }

            // Positive values keep a variance array valid
            let mut next = 0.05_f32;
            for mut entry in layer.weights_mut() {
                entry.value.mapv_inplace(|_| {
                    next += 0.01;
                    next
                });
            }
            let written: Vec<(String, Tensor)> = layer
                .weights()
                .iter()
                .map(|entry| (format!("0.{}", entry.name), entry.value.to_owned()))
                .collect();

            let input = &case.inputs[0];
            let saved = SequentialBuilder::new().add(layer).build(input).unwrap();
            for (path, value) in &written {
                assert_eq!(
                    saved.weight(path).as_ref(),
                    Some(&value.view()),
                    "{name}: the model build changed `{path}`"
                );
            }

            let file = std::env::temp_dir().join(format!(
                "rustyml_layer_contract_{}_{name}.bin",
                std::process::id()
            ));
            saved.save_to_path(&file).unwrap();
            let mut loaded = SequentialBuilder::new().add(make()).build(input).unwrap();
            let load = loaded.load_from_path(&file);
            let _ = std::fs::remove_file(&file);
            load.unwrap_or_else(|error| panic!("{name} refused its own checkpoint: {error}"));

            assert_eq!(loaded.weight_paths(), saved.weight_paths(), "{name}");
            for (path, value) in &written {
                assert_eq!(
                    loaded.weight(path).as_ref(),
                    Some(&value.view()),
                    "{name}: the load did not restore `{path}`"
                );
            }
            let x = &case.sample()[0];
            assert_eq!(
                loaded.predict(x).unwrap(),
                saved.predict(x).unwrap(),
                "{name}"
            );

            self.covered.insert(name);
        }
    }

    let mut visitor = RoundTrip {
        covered: BTreeSet::new(),
        merge: 0,
    };
    visit_every_layer(&mut visitor);
    assert_eq!(visitor.covered.len() + visitor.merge, LAYER_TYPE_COUNT);
}
