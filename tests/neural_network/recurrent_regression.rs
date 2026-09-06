//! Bit-exact regression of the 3 recurrent layers
//!
//! The golden-fixture net under [`golden`](super::golden) sees every recurrent layer with the
//! weights already set, so it never observes a draw. This file closes that gap, and it holds
//! the whole recurrent family against a table of `f32` bit patterns.
//!
//! Part 1 records what each layer draws, over 2 seeds and 2 dimension pairs. A change to the
//! order of the draws, to the fan of an initializer, or to the shape of a drawn array moves a
//! digest here and nothing else in the suite sees it.
//!
//! Part 2 trains 1 model per layer, and 1 model that stacks all 3 with `return_sequences` and
//! `go_backwards` set. That pins the time loop, the backpropagation through time, the optimizer
//! walk, and the checkpoint round trip together.
//!
//! Every recorded value is an `f32` bit pattern, so the comparison is exact and no tolerance
//! hides a change. A value here moves only when the numbers of the crate move. That is either a
//! defect or a deliberate change, and either way it must be reviewed and not silently
//! rewritten.

use super::common::GlobalSeedGuard;
use ndarray::{Array, IxDyn};
use rustyml::neural_network::Shape;
use rustyml::neural_network::layers::{Activation, Dense, GRU, LSTM, SimpleRNN};
use rustyml::neural_network::losses::mean_absolute_error::MeanAbsoluteError;
use rustyml::neural_network::losses::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::optimizers::{Adam, RMSprop, SGD};
use rustyml::neural_network::sequential::{Sequential, SequentialBuilder};
use rustyml::neural_network::traits::UnaryLayer;
use std::fmt::Write as _;

/// Builds a tensor from a pure formula, so the data never moves
fn data(shape: &[usize]) -> Array<f32, IxDyn> {
    let count: usize = shape.iter().product();
    let values: Vec<f32> = (0..count)
        .map(|i| ((((i * 37) % 101) as f32) - 50.0) / 25.0)
        .collect();
    Array::from_shape_vec(IxDyn(shape), values).expect("the formula fills the shape")
}

/// Folds every value of a tensor into 1 digest, over the bit patterns
fn digest<'a>(values: impl Iterator<Item = &'a f32>) -> u64 {
    let mut fold: u64 = 0xcbf29ce484222325;
    for value in values {
        fold ^= u64::from(value.to_bits());
        fold = fold.wrapping_mul(0x100000001b3);
    }
    fold
}

/// Prints every array of 1 built layer, in weight-path order
fn report_draw(out: &mut String, name: &str, layer: &dyn UnaryLayer) {
    for weight in layer.weights() {
        let _ = writeln!(
            out,
            "draw {name} {} shape={:?} digest={:016x}",
            weight.name,
            weight.value.shape(),
            digest(weight.value.iter())
        );
    }
}

/// Builds 1 layer against `[2, 4, input_dim]` and records what it drew
fn draws(out: &mut String) {
    for (input_dim, units) in [(4_usize, 3_usize), (3, 7)] {
        for seed in [1_u64, 12345] {
            let shape = Shape::known(&[2, 4, input_dim]);

            let mut simple = SimpleRNN::new(units, Activation::Tanh)
                .expect("units is above 0")
                .with_random_state(seed);
            simple
                .build(&shape)
                .expect("the shape names a feature count");
            report_draw(
                out,
                &format!("SimpleRNN.{input_dim}.{units}.{seed}"),
                &simple,
            );

            let mut lstm = LSTM::new(units, Activation::Tanh)
                .expect("units is above 0")
                .with_random_state(seed);
            lstm.build(&shape).expect("the shape names a feature count");
            report_draw(out, &format!("LSTM.{input_dim}.{units}.{seed}"), &lstm);

            let mut gru = GRU::new(units, Activation::Tanh)
                .expect("units is above 0")
                .with_random_state(seed);
            gru.build(&shape).expect("the shape names a feature count");
            report_draw(out, &format!("GRU.{input_dim}.{units}.{seed}"), &gru);
        }
    }
}

/// Prints every loss, every array, and the prediction of 1 trained model
fn report(
    out: &mut String,
    name: &str,
    model: &Sequential,
    history: &[f32],
    x: &Array<f32, IxDyn>,
) {
    for (epoch, loss) in history.iter().enumerate() {
        let _ = writeln!(out, "{name} loss.{epoch} {:08x}", loss.to_bits());
    }
    for path in model.weight_paths() {
        let array = model
            .weight(&path)
            .expect("the path names an array of the model");
        let _ = writeln!(
            out,
            "{name} weight {path} len={} digest={:016x}",
            array.len(),
            digest(array.iter())
        );
    }
    let prediction = model.predict(x).expect("the model accepts the input");
    let _ = writeln!(
        out,
        "{name} predict shape={:?} digest={:016x}",
        prediction.shape(),
        digest(prediction.iter())
    );
}

/// A SimpleRNN into a dense head, trained by stochastic gradient descent with momentum
fn simple_rnn_model(out: &mut String) {
    let _seed = GlobalSeedGuard::set(20260906);
    let x = data(&[6, 5, 4]);
    let y = data(&[6, 2]);

    let mut model = SequentialBuilder::new_with_seed(101)
        .add(SimpleRNN::new(4, Activation::Tanh).unwrap())
        .add(Dense::new(2, Activation::Linear).unwrap())
        .build(&Shape::known(&[6, 5, 4]))
        .unwrap();
    model.compile(
        SGD::new(0.05, 0.9, false, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    let history = model.fit(&x, &y, 3).unwrap();
    report(out, "simple", &model, history.loss(), &x);
}

/// An LSTM into a dense head, trained by Adam
fn lstm_model(out: &mut String) {
    let _seed = GlobalSeedGuard::set(20260906);
    let x = data(&[6, 5, 4]);
    let y = data(&[6, 2]);

    let mut model = SequentialBuilder::new_with_seed(202)
        .add(LSTM::new(3, Activation::Tanh).unwrap())
        .add(Dense::new(2, Activation::Linear).unwrap())
        .build(&Shape::known(&[6, 5, 4]))
        .unwrap();
    model.compile(
        Adam::new(0.01, 0.9, 0.999, 1e-8, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    let history = model.fit(&x, &y, 3).unwrap();
    report(out, "lstm", &model, history.loss(), &x);
}

/// A GRU into a dense head, trained by RMSprop against the mean absolute error
fn gru_model(out: &mut String) {
    let _seed = GlobalSeedGuard::set(20260906);
    let x = data(&[6, 5, 4]);
    let y = data(&[6, 2]);

    let mut model = SequentialBuilder::new_with_seed(303)
        .add(GRU::new(3, Activation::Tanh).unwrap())
        .add(Dense::new(2, Activation::Linear).unwrap())
        .build(&Shape::known(&[6, 5, 4]))
        .unwrap();
    model.compile(
        RMSprop::new(0.005, 0.9, 1e-8, 0.0).unwrap(),
        MeanAbsoluteError::new(),
    );
    let history = model.fit(&x, &y, 3).unwrap();
    report(out, "gru", &model, history.loss(), &x);
}

/// All 3 layers in 1 stack, with `return_sequences` and `go_backwards` set, then a checkpoint
/// round trip
///
/// The first layer returns a sequence and reads the time axis from last to first, so the stack
/// pins the sequence path, the reversed loop, and the map from a processing step back to its
/// input timestep.
fn stacked_model(out: &mut String) {
    let _seed = GlobalSeedGuard::set(20260906);
    let x = data(&[6, 5, 4]);
    let y = data(&[6, 2]);

    let mut model = SequentialBuilder::new_with_seed(404)
        .add(
            SimpleRNN::new(4, Activation::ReLU)
                .unwrap()
                .with_return_sequences(true)
                .with_go_backwards(true),
        )
        .add(
            LSTM::new(3, Activation::Tanh)
                .unwrap()
                .with_return_sequences(true),
        )
        .add(
            GRU::new(3, Activation::Tanh)
                .unwrap()
                .with_go_backwards(true),
        )
        .add(Dense::new(2, Activation::Linear).unwrap())
        .build(&Shape::known(&[6, 5, 4]))
        .unwrap();
    model.compile(
        Adam::new(0.01, 0.9, 0.999, 1e-8, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    let history = model.fit_with_batches(&x, &y, 3, 3).unwrap();
    report(out, "stack", &model, history.loss(), &x);

    let dir = std::env::temp_dir().join(format!(
        "rustyml_recurrent_regression_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).expect("the temporary directory is writable");
    let path = dir.join("stack.rustyml");
    model.save_to_path(&path).expect("the model saves");

    let mut reloaded = SequentialBuilder::new_with_seed(999)
        .add(
            SimpleRNN::new(4, Activation::ReLU)
                .unwrap()
                .with_return_sequences(true)
                .with_go_backwards(true),
        )
        .add(
            LSTM::new(3, Activation::Tanh)
                .unwrap()
                .with_return_sequences(true),
        )
        .add(
            GRU::new(3, Activation::Tanh)
                .unwrap()
                .with_go_backwards(true),
        )
        .add(Dense::new(2, Activation::Linear).unwrap())
        .build(&Shape::known(&[6, 5, 4]))
        .unwrap();
    reloaded.load_from_path(&path).expect("the model loads");
    let _ = std::fs::remove_dir_all(&dir);

    let prediction = reloaded.predict(&x).expect("the model accepts the input");
    let _ = writeln!(
        out,
        "stack reload predict shape={:?} digest={:016x}",
        prediction.shape(),
        digest(prediction.iter())
    );
}

/// Renders every case into 1 table
fn render() -> String {
    let mut out = String::new();
    draws(&mut out);
    simple_rnn_model(&mut out);
    lstm_model(&mut out);
    gru_model(&mut out);
    stacked_model(&mut out);
    out
}

/// The recorded table. Every line is an `f32` bit pattern or a digest over bit patterns
const EXPECTED: &str = "\
draw SimpleRNN.4.3.1 kernel shape=[4, 3] digest=e855140784a925bc\n\
draw SimpleRNN.4.3.1 recurrent_kernel shape=[3, 3] digest=f6757293047b0430\n\
draw SimpleRNN.4.3.1 bias shape=[1, 3] digest=d94d12186c0f2fb7\n\
draw LSTM.4.3.1 kernel shape=[4, 12] digest=9baa838d71ad1617\n\
draw LSTM.4.3.1 recurrent_kernel shape=[3, 12] digest=5cb83473b71a556e\n\
draw LSTM.4.3.1 bias shape=[1, 12] digest=c0e812a43b906495\n\
draw GRU.4.3.1 kernel shape=[4, 9] digest=cccd497ba8428be4\n\
draw GRU.4.3.1 recurrent_kernel shape=[3, 9] digest=c0b073eb58e8abf2\n\
draw GRU.4.3.1 bias shape=[1, 9] digest=e604823a249029bf\n\
draw SimpleRNN.4.3.12345 kernel shape=[4, 3] digest=4d5cb9cc2b75f84d\n\
draw SimpleRNN.4.3.12345 recurrent_kernel shape=[3, 3] digest=6b991f4dafb6deed\n\
draw SimpleRNN.4.3.12345 bias shape=[1, 3] digest=d94d12186c0f2fb7\n\
draw LSTM.4.3.12345 kernel shape=[4, 12] digest=a71cc05416b7bdc7\n\
draw LSTM.4.3.12345 recurrent_kernel shape=[3, 12] digest=7e4facdc1d123545\n\
draw LSTM.4.3.12345 bias shape=[1, 12] digest=c0e812a43b906495\n\
draw GRU.4.3.12345 kernel shape=[4, 9] digest=b95d0d7259a70ac3\n\
draw GRU.4.3.12345 recurrent_kernel shape=[3, 9] digest=96e72aa14440394d\n\
draw GRU.4.3.12345 bias shape=[1, 9] digest=e604823a249029bf\n\
draw SimpleRNN.3.7.1 kernel shape=[3, 7] digest=4b8bfd84eb3d9a6a\n\
draw SimpleRNN.3.7.1 recurrent_kernel shape=[7, 7] digest=dba07ac98686fa89\n\
draw SimpleRNN.3.7.1 bias shape=[1, 7] digest=778b1a14b6876aa7\n\
draw LSTM.3.7.1 kernel shape=[3, 28] digest=edb1e5750aaaf479\n\
draw LSTM.3.7.1 recurrent_kernel shape=[7, 28] digest=0382a44b8441a498\n\
draw LSTM.3.7.1 bias shape=[1, 28] digest=5ebaf39ed65081d5\n\
draw GRU.3.7.1 kernel shape=[3, 21] digest=0b71812977e051a1\n\
draw GRU.3.7.1 recurrent_kernel shape=[7, 21] digest=079a5b241374def2\n\
draw GRU.3.7.1 bias shape=[1, 21] digest=98b2b1418e80a50f\n\
draw SimpleRNN.3.7.12345 kernel shape=[3, 7] digest=c5d3ffabb7ac6bea\n\
draw SimpleRNN.3.7.12345 recurrent_kernel shape=[7, 7] digest=e15acd9afee550d0\n\
draw SimpleRNN.3.7.12345 bias shape=[1, 7] digest=778b1a14b6876aa7\n\
draw LSTM.3.7.12345 kernel shape=[3, 28] digest=a1c3b1304dadfc60\n\
draw LSTM.3.7.12345 recurrent_kernel shape=[7, 28] digest=018226ff6eb40760\n\
draw LSTM.3.7.12345 bias shape=[1, 28] digest=5ebaf39ed65081d5\n\
draw GRU.3.7.12345 kernel shape=[3, 21] digest=68f3b471f893bf41\n\
draw GRU.3.7.12345 recurrent_kernel shape=[7, 21] digest=439f875960649c5a\n\
draw GRU.3.7.12345 bias shape=[1, 21] digest=98b2b1418e80a50f\n\
simple loss.0 3fc76714\n\
simple loss.1 3f9a715f\n\
simple loss.2 3f5e12df\n\
simple weight 0.kernel len=16 digest=51aac79d3e365cb8\n\
simple weight 0.recurrent_kernel len=16 digest=c4aebc280b9ad5ab\n\
simple weight 0.bias len=4 digest=b7095315f2b5feaf\n\
simple weight 1.kernel len=8 digest=01f65d776773d3ed\n\
simple weight 1.bias len=2 digest=4162df683ef9fda9\n\
simple predict shape=[6, 2] digest=8715f91b1074d5d1\n\
lstm loss.0 3fcea85f\n\
lstm loss.1 3fc7a289\n\
lstm loss.2 3fc0d3f4\n\
lstm weight 0.kernel len=48 digest=3b2e9e3607d95d97\n\
lstm weight 0.recurrent_kernel len=36 digest=8470e6fd6805a706\n\
lstm weight 0.bias len=12 digest=45ab8280529c57c8\n\
lstm weight 1.kernel len=6 digest=f8550f55d0105f4b\n\
lstm weight 1.bias len=2 digest=ff40adeeaf5a6186\n\
lstm predict shape=[6, 2] digest=e193be1a2bb38a1f\n\
gru loss.0 3f95ce0e\n\
gru loss.1 3f9010c9\n\
gru loss.2 3f8bc331\n\
gru weight 0.kernel len=36 digest=024bb3c96fd02452\n\
gru weight 0.recurrent_kernel len=27 digest=17355462c83d4245\n\
gru weight 0.bias len=9 digest=0b26abbdacff3757\n\
gru weight 1.kernel len=6 digest=dab7a39c1249f341\n\
gru weight 1.bias len=2 digest=1d1c198867745303\n\
gru predict shape=[6, 2] digest=594f5f46b6a30ab5\n\
stack loss.0 3fd4a1f2\n\
stack loss.1 3fcb9ac4\n\
stack loss.2 3fc327f6\n\
stack weight 0.kernel len=16 digest=fa3df18c3b687d2e\n\
stack weight 0.recurrent_kernel len=16 digest=a6de27897532f391\n\
stack weight 0.bias len=4 digest=1a96320545b4a2eb\n\
stack weight 1.kernel len=48 digest=652e407c1f62e082\n\
stack weight 1.recurrent_kernel len=36 digest=4499f4807dba5ee0\n\
stack weight 1.bias len=12 digest=9c30a3c0b2e221b6\n\
stack weight 2.kernel len=27 digest=82308c0c8ace0b61\n\
stack weight 2.recurrent_kernel len=27 digest=dde3cd9f870b052f\n\
stack weight 2.bias len=9 digest=ef94d34861d50bd5\n\
stack weight 3.kernel len=6 digest=5fac098d65d7d557\n\
stack weight 3.bias len=2 digest=c054e4750e4e5058\n\
stack predict shape=[6, 2] digest=46662fc879c1252d\n\
stack reload predict shape=[6, 2] digest=46662fc879c1252d\n";

/// The recurrent family gives the same bits it gave when the table was captured
#[test]
fn the_recurrent_family_holds_its_recorded_bits() {
    let got = render();
    if got != EXPECTED {
        let expected_lines: Vec<&str> = EXPECTED.lines().collect();
        let got_lines: Vec<&str> = got.lines().collect();
        let mut differences = String::new();
        for index in 0..expected_lines.len().max(got_lines.len()) {
            let expected = expected_lines.get(index).copied().unwrap_or("<missing>");
            let actual = got_lines.get(index).copied().unwrap_or("<missing>");
            if expected != actual {
                let _ = writeln!(
                    differences,
                    "line {index}:\n  want {expected}\n  got  {actual}"
                );
            }
        }
        panic!("the recurrent family moved:\n{differences}\nfull output:\n{got}");
    }
}
