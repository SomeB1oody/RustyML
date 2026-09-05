//! Bit-exact regression of a whole trained model
//!
//! The golden-fixture net under [`golden`](super::golden) sees 1 layer at a time and never
//! sees a model. This file is the guard for the layer above it: the training loop, the
//! optimizer walk, the loss, the batch shuffle, the running statistics of a normalization
//! layer, the random stream of a dropout layer, and the checkpoint round trip.
//!
//! Every recorded value is an `f32` bit pattern, so the comparison is exact and no tolerance
//! hides a change. The table was captured from the code as it stood BEFORE the call contract
//! moved every cache and every gradient into a
//! [`Ctx`](rustyml::neural_network::Ctx), and it did not move by 1 bit across that change.
//!
//! A value here moves only when the numbers of the crate move. That is either a defect or a
//! deliberate change, and either way it must be reviewed and not silently rewritten.

use super::common::GlobalSeedGuard;
use ndarray::{Array, IxDyn};
use rustyml::neural_network::Shape;
use rustyml::neural_network::layers::{
    Activation, BatchNormalization, Conv2D, Dense, Dropout, Embedding, Flatten, GaussianNoise,
    GroupNormalization, LSTM, LayerNormalization, MaxPooling2D, PReLU, SimpleRNN,
};
use rustyml::neural_network::losses::categorical_cross_entropy::CategoricalCrossEntropy;
use rustyml::neural_network::losses::mean_absolute_error::MeanAbsoluteError;
use rustyml::neural_network::losses::mean_squared_error::MeanSquaredError;
use rustyml::neural_network::optimizers::{Adam, AdamW, RMSprop, SGD};
use rustyml::neural_network::sequential::{Sequential, SequentialBuilder};
use std::fmt::Write as _;

/// Builds a tensor from a pure formula, so the data never moves
fn data(shape: &[usize]) -> Array<f32, IxDyn> {
    let count: usize = shape.iter().product();
    let values: Vec<f32> = (0..count)
        .map(|i| ((((i * 37) % 101) as f32) - 50.0) / 25.0)
        .collect();
    Array::from_shape_vec(IxDyn(shape), values).expect("the formula fills the shape")
}

/// Builds a tensor of whole-number indices below `limit`
fn indices(shape: &[usize], limit: usize) -> Array<f32, IxDyn> {
    let count: usize = shape.iter().product();
    let values: Vec<f32> = (0..count).map(|i| ((i * 7) % limit) as f32).collect();
    Array::from_shape_vec(IxDyn(shape), values).expect("the formula fills the shape")
}

/// Builds a 1-hot target of `classes` columns
fn one_hot(rows: usize, classes: usize) -> Array<f32, IxDyn> {
    let mut values = vec![0.0_f32; rows * classes];
    for row in 0..rows {
        values[row * classes + (row * 3) % classes] = 1.0;
    }
    Array::from_shape_vec(IxDyn(&[rows, classes]), values).expect("the shape holds every value")
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

/// A convolution stack with batch normalization and dropout, trained by Adam
fn convolution(out: &mut String) {
    let _seed = GlobalSeedGuard::set(20260904);
    let x = data(&[8, 6, 6, 2]);
    let y = data(&[8, 3]);

    let mut model = SequentialBuilder::new_with_seed(777)
        .add(Conv2D::new(4, (3, 3), (1, 1), Activation::ReLU).unwrap())
        .add(BatchNormalization::new(0.9, 1e-5).unwrap())
        .add(GroupNormalization::new(2, 1e-5).unwrap())
        .add(MaxPooling2D::new((2, 2)))
        .add(Flatten::new())
        .add(Dropout::new(0.25).unwrap())
        .add(Dense::new(5, Activation::Tanh).unwrap())
        .add(Dense::new(3, Activation::Linear).unwrap())
        .build(&Shape::known(&[8, 6, 6, 2]))
        .unwrap();
    model.compile(
        Adam::new(0.01, 0.9, 0.999, 1e-8, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let history = model.fit_with_batches(&x, &y, 4, 4).unwrap();
    report(out, "conv", &model, history.loss(), &x);
    let _ = writeln!(
        out,
        "conv evaluate {:08x}",
        model.evaluate(&x, &y).unwrap().to_bits()
    );
}

/// A recurrent stack trained by SGD with momentum and clip-by-global-norm
fn recurrent(out: &mut String) {
    let _seed = GlobalSeedGuard::set(31415926);
    let x = data(&[6, 5, 4]);
    let y = data(&[6, 2]);

    let mut model = SequentialBuilder::new_with_seed(2718)
        .add(SimpleRNN::new(4, Activation::Tanh).unwrap())
        .add(LayerNormalization::new(1e-5).unwrap())
        .add(Dense::new(2, Activation::Linear).unwrap())
        .build(&Shape::known(&[6, 5, 4]))
        .unwrap();
    model.compile(
        SGD::new(0.05, 0.9, true, 0.0)
            .unwrap()
            .with_global_clipnorm(1.0)
            .unwrap(),
        MeanAbsoluteError::new(),
    );

    let history = model.fit(&x, &y, 5).unwrap();
    report(out, "rnn", &model, history.loss(), &x);
}

/// A gated recurrent stack trained by RMSprop
fn gated(out: &mut String) {
    let _seed = GlobalSeedGuard::set(161803);
    let x = data(&[4, 6, 3]);
    let y = data(&[4, 3]);

    let mut model = SequentialBuilder::new_with_seed(577)
        .add(LSTM::new(3, Activation::Tanh).unwrap())
        .add(BatchNormalization::new(0.7, 1e-5).unwrap())
        .add(PReLU::new(0.2).unwrap())
        .add(Dense::new(3, Activation::Linear).unwrap())
        .build(&Shape::known(&[4, 6, 3]))
        .unwrap();
    model.compile(
        RMSprop::new(0.01, 0.9, 1e-7, 0.0).unwrap(),
        MeanSquaredError::new(),
    );

    let history = model.fit(&x, &y, 4).unwrap();
    report(out, "lstm", &model, history.loss(), &x);
}

/// An embedding stack with noise, trained by AdamW against cross-entropy
fn embedding(out: &mut String) {
    let _seed = GlobalSeedGuard::set(112358);
    let x = indices(&[6, 4], 10);
    let y = one_hot(6, 4);

    let mut model = SequentialBuilder::new_with_seed(1123)
        .add(Embedding::new(10, 3).unwrap())
        .add(GaussianNoise::new(0.1).unwrap())
        .add(Flatten::new())
        .add(Dense::new(4, Activation::Softmax { axis: -1 }).unwrap())
        .build(&Shape::known(&[6, 4]))
        .unwrap();
    model.compile(
        AdamW::new(0.02, 0.9, 0.999, 1e-8, 0.01).unwrap(),
        CategoricalCrossEntropy::new(false),
    );

    let history = model.fit_with_batches(&x, &y, 3, 3).unwrap();
    report(out, "embed", &model, history.loss(), &x);
}

/// A checkpoint round trip, which must give back the same prediction
fn round_trip(out: &mut String) {
    let _seed = GlobalSeedGuard::set(999983);
    let x = data(&[4, 5]);
    let y = data(&[4, 2]);

    let mut model = SequentialBuilder::new_with_seed(31)
        .add(Dense::new(6, Activation::ReLU).unwrap())
        .add(BatchNormalization::new(0.8, 1e-4).unwrap())
        .add(Dense::new(2, Activation::Linear).unwrap())
        .build(&Shape::known(&[4, 5]))
        .unwrap();
    model.compile(
        Adam::new(0.01, 0.9, 0.999, 1e-8, 0.0).unwrap(),
        MeanSquaredError::new(),
    );
    model.fit(&x, &y, 3).unwrap();

    let path = std::env::temp_dir().join("rustyml_model_baseline.bin");
    model.save_to_path(&path).unwrap();

    let mut loaded = SequentialBuilder::new_with_seed(31)
        .add(Dense::new(6, Activation::ReLU).unwrap())
        .add(BatchNormalization::new(0.8, 1e-4).unwrap())
        .add(Dense::new(2, Activation::Linear).unwrap())
        .build(&Shape::known(&[4, 5]))
        .unwrap();
    loaded.load_from_path(&path).unwrap();
    std::fs::remove_file(&path).unwrap();

    let before = model.predict(&x).unwrap();
    let after = loaded.predict(&x).unwrap();
    let _ = writeln!(out, "roundtrip before={:016x}", digest(before.iter()));
    let _ = writeln!(out, "roundtrip after={:016x}", digest(after.iter()));
}

/// Every value the 5 models above produce, as the code produced them before the call
/// contract changed
const EXPECTED: &str = "\
conv loss.0 402ba480\n\
conv loss.1 4015700c\n\
conv loss.2 3ffa72c8\n\
conv loss.3 3fdcc1d5\n\
conv weight 0.kernel len=72 digest=33f158acc29da6f6\n\
conv weight 0.bias len=4 digest=cf92edb2e091004c\n\
conv weight 1.gamma len=4 digest=f397779fbc721258\n\
conv weight 1.beta len=4 digest=1fd364e801b3bd55\n\
conv weight 1.moving_mean len=4 digest=c2fa27e2e5b96f5c\n\
conv weight 1.moving_variance len=4 digest=f458c29fc7d64e79\n\
conv weight 2.gamma len=4 digest=05a45bafcb270146\n\
conv weight 2.beta len=4 digest=335e84954e4e3ba4\n\
conv weight 6.kernel len=80 digest=0e1cd78341718164\n\
conv weight 6.bias len=5 digest=d364ac5e67f50a30\n\
conv weight 7.kernel len=15 digest=04b10614a2e82266\n\
conv weight 7.bias len=3 digest=18ca44a095c83f18\n\
conv predict shape=[8, 3] digest=8897ef36d6990fa7\n\
conv evaluate 3fb8045f\n\
rnn loss.0 3fd31f2f\n\
rnn loss.1 3fc448a8\n\
rnn loss.2 3fa94725\n\
rnn loss.3 3f8290b0\n\
rnn loss.4 3f3eb488\n\
rnn weight 0.kernel len=16 digest=29fdbbc711517335\n\
rnn weight 0.recurrent_kernel len=16 digest=92b7aac8bac63adf\n\
rnn weight 0.bias len=4 digest=49e0be6320544eb9\n\
rnn weight 1.gamma len=4 digest=ba63537f90cea3a5\n\
rnn weight 1.beta len=4 digest=19e641e75f0bc5d7\n\
rnn weight 2.kernel len=8 digest=34888bd054e7c780\n\
rnn weight 2.bias len=2 digest=0d15a987795a6976\n\
rnn predict shape=[6, 2] digest=5d447a92341bc144\n\
lstm loss.0 3fb7d5d8\n\
lstm loss.1 3f9dfcfa\n\
lstm loss.2 3f91b02a\n\
lstm loss.3 3f8818c7\n\
lstm weight 0.kernel len=36 digest=17aaefe8724df499\n\
lstm weight 0.recurrent_kernel len=36 digest=98005fe2998a28e0\n\
lstm weight 0.bias len=12 digest=9f5573b3e9ce60e1\n\
lstm weight 1.gamma len=3 digest=87640972944bc70f\n\
lstm weight 1.beta len=3 digest=c3b8d8b4fb478d06\n\
lstm weight 1.moving_mean len=3 digest=8e94af3c8e1d6cbc\n\
lstm weight 1.moving_variance len=3 digest=d673d5cdad6174f7\n\
lstm weight 2.alpha len=3 digest=12396be5484816a9\n\
lstm weight 3.kernel len=9 digest=e11c9accb2c35609\n\
lstm weight 3.bias len=3 digest=622168849ed2ded9\n\
lstm predict shape=[4, 3] digest=627a647a48129e72\n\
embed loss.0 3fad2b80\n\
embed loss.1 3fa4e9e1\n\
embed loss.2 3fa8d114\n\
embed weight 0.embeddings len=30 digest=d885fe68eed9ebbe\n\
embed weight 3.kernel len=48 digest=5ad656f43bfb4a4e\n\
embed weight 3.bias len=4 digest=8c63195f8bd0cd04\n\
embed predict shape=[6, 4] digest=37b3fb11d3f5274d\n\
roundtrip before=7031cefeadc80929\n\
roundtrip after=7031cefeadc80929";

/// Trains 5 models and compares every value against the recorded table
///
/// The test runs the 5 models in 1 function on purpose. Each one sets the global seed, and
/// that seed is a process-wide value, so 2 of these running at once would race
#[test]
fn every_model_matches_the_recorded_table() {
    let mut out = String::new();
    convolution(&mut out);
    recurrent(&mut out);
    gated(&mut out);
    embedding(&mut out);
    round_trip(&mut out);

    if out.trim_end() != EXPECTED.trim_end() {
        let recorded: Vec<&str> = EXPECTED.trim_end().lines().collect();
        let produced: Vec<&str> = out.trim_end().lines().collect();
        let mut report = String::from("the model regression table moved:\n");
        for index in 0..recorded.len().max(produced.len()) {
            let was = recorded.get(index).copied().unwrap_or("<absent>");
            let now = produced.get(index).copied().unwrap_or("<absent>");
            if was != now {
                report.push_str(&format!(
                    "  line {index}: recorded `{was}`, produced `{now}`\n"
                ));
            }
        }
        panic!("{report}");
    }
}
