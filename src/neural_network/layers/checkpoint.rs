//! The named checkpoint format for a Sequential model
//!
//! A checkpoint addresses every array of a model by a dotted path. The path is
//! `<scope>.<name>`: the position of the layer, counted from the input, and the name that the
//! layer gives the array. It is the same pair that
//! [`ParamId`](crate::neural_network::traits::ParamId) uses for optimizer state, so 1 address
//! serves the training loop and the file alike
//!
//! # What a file holds
//!
//! ```text
//! ModelCheckpoint      magic, format_version, layers
//! LayerCheckpoint      layer_type, build, weights
//! WeightRecord         name, kind, shape, data
//! ```
//!
//! The layer type is the string that [`Layer::layer_type`] returns, and a load compares it
//! per position. Name and shape alone are too weak for that comparison:
//! `InstanceNormalization`, `GroupNormalization`, and a rank-2 `LayerNormalization` all hold
//! `gamma` and `beta` of the same extent, so nothing but the type name tells them apart
//!
//! The kind of a record says whether an optimizer updates the array. It separates the
//! `moving_mean` and the `moving_variance` of
//! [`BatchNormalization`](crate::neural_network::layers::regularization::normalization::batch_normalization::BatchNormalization)
//! from the trainable `gamma` and `beta` that stand next to them in the same layer
//!
//! # Strict is the default
//!
//! [`apply`] refuses a file that disagrees with the model in any way, and the refusal names
//! the parameter path. [`apply_partial`] applies what matches and reports the rest, and a
//! caller asks for it by name. A lenient default would load a file that is wrong for the model
//! and leave the difference to show up as a wrong prediction
//!
//! [`Layer::layer_type`]: crate::neural_network::traits::Layer::layer_type
//! [`apply`]: crate::neural_network::layers::checkpoint::apply
//! [`apply_partial`]: crate::neural_network::layers::checkpoint::apply_partial
//! [`Layer::build`]: crate::neural_network::traits::Layer::build

use crate::error::{Error, IoError};
use crate::neural_network::Shape;
use crate::neural_network::traits::{Layer, WeightKind};
use crate::{Deserialize, Serialize};
use ndarray::ArrayViewMutD;
use std::borrow::Cow;

/// Magic tag at the head of every saved model (`"RMLM"` in ASCII)
///
/// postcard is not self-describing, so without a tag a file from an older release is *parsed*
/// as layer data rather than refused. A file that does not start with this is either not a
/// RustyML model or predates the versioned format
pub const MODEL_MAGIC: u32 = 0x524D_4C4D;

/// On-disk model format version written by this build
///
/// Version 2 is the named checkpoint. It addresses every array by `<scope>.<name>`, it carries
/// the kind of each array, and it reserves the build slot. Version 1 held a closed enum of
/// per-layer weight containers, whose variant index it wrote in place of any name. No byte of
/// the 2 layouts agrees, so every version 1 file stops loading, and the refusal names both
/// versions
///
/// Bump this on any change to the layout of a record, to the order of the fields of a
/// structure, or to the meaning of a field. The load path checks the layer count, the layer
/// type of each position, and the name, the kind, and the shape of every array. Those checks
/// can all pass for a file that another release wrote, so this number is what makes such a
/// file fail instead of loading values that mean something else
pub const MODEL_FORMAT_VERSION: u32 = 2;

/// The shape that a layer was built for
///
/// A layer allocates every array it owns in
/// [`Layer::build`], from the shape of its input.
/// The build shape is therefore the 1 thing a fresh layer does not have, and it decides every
/// extent the layer allocates. A file carries it so that a load can refuse a model that was
/// built for another input
///
/// The batch axis is free. A layer serves every batch size, so the batch extent is not part of
/// what the layer was built for, and a model built for 32 samples takes the checkpoint of a
/// model built for 1
///
/// A load compares the field when the file and the layer both carry one, and skips the
/// comparison in every other case. A layer that owns no array and reads no extent of its
/// input, such as an activation, carries none
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Shape of the input the layer was built for, batch axis first and free
    pub input_shape: Shape,
}

impl BuildConfig {
    /// The build record of a layer that built for `input`
    ///
    /// The batch axis is freed here, so every caller records the same canonical form
    ///
    /// # Parameters
    ///
    /// - `input` - Shape the layer built for, batch axis first
    ///
    /// # Returns
    ///
    /// - `BuildConfig` - The record, with a free batch axis
    #[inline]
    pub fn new(input: &Shape) -> Self {
        Self {
            input_shape: input.free_batch(),
        }
    }
}

/// 1 named array of 1 layer, as a file holds it
///
/// The `'a` lifetime is threaded from the layer: saving borrows the live array and the live
/// name, and loading owns both
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightRecord<'a> {
    /// Name the layer gives the array. It is the second half of the checkpoint path
    pub name: Cow<'a, str>,
    /// Whether an optimizer updates the array
    pub kind: WeightKind,
    /// Shape of the array, in C order
    pub shape: Vec<usize>,
    /// Every element of the array, in logical C order
    pub data: Cow<'a, [f32]>,
}

/// 1 layer of a model, as a file holds it
///
/// The `'a` lifetime is threaded from the layer, as in [`WeightRecord`]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerCheckpoint<'a> {
    /// The string that [`Layer::layer_type`] returned. A load compares it against the layer
    /// at the same position
    pub layer_type: Cow<'a, str>,
    /// The shape the layer was built for, when the layer reports one. See [`BuildConfig`]
    pub build: Option<BuildConfig>,
    /// Every array the layer holds, in the order [`Layer::weights`] gives them
    pub weights: Vec<WeightRecord<'a>>,
}

/// A whole model, as a file holds it
///
/// The `'a` lifetime is threaded from the layers, as in [`WeightRecord`]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCheckpoint<'a> {
    /// Magic tag identifying a RustyML model file. See [`MODEL_MAGIC`]
    ///
    /// This field comes first, so a file written before this header existed misreads its
    /// leading layer count as the tag. Load then rejects it before it can apply any weights
    pub magic: u32,
    /// On-disk format version of this file. See [`MODEL_FORMAT_VERSION`]
    pub format_version: u32,
    /// Every layer, from the input
    pub layers: Vec<LayerCheckpoint<'a>>,
}

/// What a lenient load did, and what it left undone
///
/// [`apply_partial`] fills it. Every entry is a checkpoint path, and the 3 lists together
/// cover every path of the model and every path of the file
///
/// An array whose shape or kind disagrees is in `missing` and in `unused` at the same time.
/// The model did not get a value for it, and the file value went nowhere
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct LoadReport {
    /// Paths that the load wrote into the model
    pub applied: Vec<String>,
    /// Paths of the model that got no value
    pub missing: Vec<String>,
    /// Paths of the file that reached no array
    pub unused: Vec<String>,
}

/// Builds the checkpoint path of 1 array
///
/// # Parameters
///
/// - `scope` - Position of the layer in the model, counted from the input
/// - `name` - Name the layer gives the array
///
/// # Returns
///
/// - `String` - The dotted path, such as `2.kernel`
#[inline]
pub fn weight_path(scope: usize, name: &str) -> String {
    format!("{scope}.{name}")
}

/// Reads every array of every layer into a checkpoint that borrows the model
///
/// Nothing is copied. Each record borrows the name and the elements of the live array, so the
/// checkpoint is a view of the model until it is serialized. An array that a layer does not
/// hold in 1 contiguous run is the 1 exception, and its record owns a C-order copy
///
/// # Parameters
///
/// - `layers` - The layers of the model, from the input
///
/// # Returns
///
/// - `ModelCheckpoint` - The whole model, with the magic tag and the format version of this
///   build
pub fn capture(layers: &[Box<dyn Layer>]) -> ModelCheckpoint<'_> {
    let recorded = layers
        .iter()
        .map(|layer| LayerCheckpoint {
            layer_type: Cow::Borrowed(layer.layer_type()),
            build: layer.build_config(),
            weights: layer
                .weights()
                .into_iter()
                .map(|entry| WeightRecord {
                    name: Cow::Borrowed(entry.name),
                    kind: entry.kind,
                    shape: entry.value.shape().to_vec(),
                    // `to_slice` keeps the lifetime of the layer, so a C-order array rides
                    // into the file with no copy at all
                    data: match entry.value.to_slice() {
                        Some(run) => Cow::Borrowed(run),
                        None => Cow::Owned(entry.value.iter().copied().collect()),
                    },
                })
                .collect(),
        })
        .collect();

    ModelCheckpoint {
        magic: MODEL_MAGIC,
        format_version: MODEL_FORMAT_VERSION,
        layers: recorded,
    }
}

/// Applies a checkpoint to a model, and refuses every disagreement
///
/// The load runs in 2 passes, and the first one writes nothing. It checks, in this order:
///
/// 1. The number of layers.
/// 2. The layer type of each position.
/// 3. The build shape of a position, when the file and the layer both carry one.
/// 4. The number of arrays of a layer, and the name of each one, in order.
/// 5. The kind of each array.
/// 6. The shape of each array, and the element count of the record.
///
/// The second pass writes. A refusal therefore leaves the model exactly as it was, and a model
/// never holds the arrays of 1 file next to the arrays of another
///
/// # Parameters
///
/// - `layers` - The layers of the model, from the input
/// - `file` - The checkpoint to apply
///
/// # Returns
///
/// - `Result<(), Error>` - Ok when every array of the model took its value
///
/// # Errors
///
/// - `Error::Io(IoError::ModelStructureMismatch)` - The file and the model disagree. The
///   message names the checkpoint path, or the layer position for a whole-layer disagreement
pub fn apply(layers: &mut [Box<dyn Layer>], file: &ModelCheckpoint<'_>) -> Result<(), Error> {
    if file.layers.len() != layers.len() {
        return Err(mismatch(format!(
            "layer count mismatch: model has {} layers, file has {} layers",
            layers.len(),
            file.layers.len()
        )));
    }

    // Pass 1 reads the model alone, so a refusal here leaves every array where it was
    for (scope, (layer, saved)) in layers.iter().zip(file.layers.iter()).enumerate() {
        let layer_type = layer.layer_type();
        if layer_type != saved.layer_type {
            return Err(mismatch(format!(
                "layer {scope} type mismatch: model has `{layer_type}`, file has `{}`",
                saved.layer_type
            )));
        }

        // A layer that owns no array carries no build shape, so this compares only when both
        // sides carry one
        if let (Some(wanted), Some(found)) = (layer.build_config(), saved.build.as_ref())
            && wanted != *found
        {
            return Err(mismatch(format!(
                "layer {scope} (`{layer_type}`) was built for input shape {}, and the file \
                 records {}",
                wanted.input_shape, found.input_shape
            )));
        }

        let targets = layer.weights();
        if targets.len() != saved.weights.len() {
            // Name both rosters. An optional parameter such as a bias is exactly the case
            // where the counts differ, and the names say which array is the extra one
            return Err(mismatch(format!(
                "layer {scope} (`{layer_type}`) holds the arrays {:?}, and the file holds {:?}",
                targets.iter().map(|e| e.name).collect::<Vec<_>>(),
                saved.weights.iter().map(|r| &*r.name).collect::<Vec<_>>()
            )));
        }

        for (target, record) in targets.iter().zip(saved.weights.iter()) {
            let path = weight_path(scope, target.name);
            if target.name != record.name {
                return Err(mismatch(format!(
                    "the model holds `{path}` where the file holds `{}`",
                    weight_path(scope, &record.name)
                )));
            }
            if target.kind != record.kind {
                return Err(mismatch(format!(
                    "`{path}` is {} in the model, and {} in the file",
                    kind_word(target.kind),
                    kind_word(record.kind)
                )));
            }
            if target.value.shape() != record.shape.as_slice() {
                return Err(mismatch(format!(
                    "`{path}` has shape {:?} in the model, and {:?} in the file",
                    target.value.shape(),
                    record.shape
                )));
            }
            if record.data.len() != target.value.len() {
                return Err(mismatch(format!(
                    "`{path}` holds {} elements at shape {:?}, and the file record carries {}",
                    target.value.len(),
                    record.shape,
                    record.data.len()
                )));
            }
        }
    }

    // Pass 2 writes, and every check above already passed
    for (layer, saved) in layers.iter_mut().zip(file.layers.iter()) {
        for (target, record) in layer.weights_mut().iter_mut().zip(saved.weights.iter()) {
            write_record(&mut target.value, &record.data);
        }
    }

    Ok(())
}

/// Applies what the file and the model agree on, and reports the rest
///
/// This is the opt-in lenient load. It writes an array when the position holds the same layer
/// type, when the 2 sides agree on the build shape, and when the file holds the same name, the
/// same kind, and the same shape. Everything else goes into the report and nothing else fails.
/// A position whose layer type differs contributes every path of that layer to both lists,
/// because a name and a shape cannot tell 2 normalization layers apart
///
/// A position whose build shape differs does the same. [`apply`] refuses such a file, and this
/// skips the layer: the 2 paths agree that a layer built for another input takes no weights.
/// The per-array shape check cannot stand in for it, because a convolution kernel is the same
/// shape for every spatial extent
///
/// # Parameters
///
/// - `layers` - The layers of the model, from the input
/// - `file` - The checkpoint to apply
///
/// # Returns
///
/// - `LoadReport` - The paths that took a value, the paths of the model that got none, and the
///   paths of the file that reached no array
pub fn apply_partial(layers: &mut [Box<dyn Layer>], file: &ModelCheckpoint<'_>) -> LoadReport {
    let mut report = LoadReport::default();

    for (scope, layer) in layers.iter_mut().enumerate() {
        let at_scope = file.layers.get(scope);
        let saved = at_scope.filter(|saved| saved.layer_type == layer.layer_type());
        let Some(saved) = saved else {
            // The position holds another layer type, or the file is shorter than the model.
            // Neither side reaches the other, so every path of both sides is left over
            report
                .missing
                .extend(layer.weights().iter().map(|e| weight_path(scope, e.name)));
            if let Some(other) = at_scope {
                report
                    .unused
                    .extend(other.weights.iter().map(|r| weight_path(scope, &r.name)));
            }
            continue;
        };

        // The lenient load skips a whole layer whose build shape disagrees, rather than
        // writing weights into a layer that was built for another input. A conv kernel does
        // not change with the spatial extents, so the per-array shape check below cannot see
        // such a disagreement. Nothing fails here: every path of both sides is reported
        if let (Some(wanted), Some(found)) = (layer.build_config(), saved.build.as_ref())
            && wanted != *found
        {
            report
                .missing
                .extend(layer.weights().iter().map(|e| weight_path(scope, e.name)));
            report
                .unused
                .extend(saved.weights.iter().map(|r| weight_path(scope, &r.name)));
            continue;
        }

        let mut taken = vec![false; saved.weights.len()];
        for target in layer.weights_mut().iter_mut() {
            let path = weight_path(scope, target.name);
            // A record whose element count contradicts its own shape reaches no array either
            let found = saved.weights.iter().position(|record| {
                record.name == target.name
                    && record.kind == target.kind
                    && record.shape.as_slice() == target.value.shape()
                    && record.data.len() == target.value.len()
            });
            match found {
                Some(index) => {
                    taken[index] = true;
                    write_record(&mut target.value, &saved.weights[index].data);
                    report.applied.push(path);
                }
                None => report.missing.push(path),
            }
        }

        report.unused.extend(
            saved
                .weights
                .iter()
                .zip(taken.iter())
                .filter(|(_, used)| !**used)
                .map(|(record, _)| weight_path(scope, &record.name)),
        );
    }

    // A file longer than the model has whole layers that reached nothing
    for (scope, saved) in file.layers.iter().enumerate().skip(layers.len()) {
        report
            .unused
            .extend(saved.weights.iter().map(|r| weight_path(scope, &r.name)));
    }

    report
}

/// Writes the elements of 1 record into the array of the layer
///
/// The caller has already compared the shape and the element count. The target keeps the memory
/// order that the layer gave it, so a load never puts a foreign layout in a layer
///
/// # Parameters
///
/// - `target` - The array of the layer, as a writable view
/// - `data` - The elements of the record, in logical C order
fn write_record(target: &mut ArrayViewMutD<'_, f32>, data: &[f32]) {
    match target.as_slice_mut() {
        // A C-order array takes the run in 1 copy, and every layer holds C order
        Some(run) => run.copy_from_slice(data),
        None => {
            for (slot, value) in target.iter_mut().zip(data.iter()) {
                *slot = *value;
            }
        }
    }
}

/// The word a message uses for 1 weight kind
fn kind_word(kind: WeightKind) -> &'static str {
    match kind {
        WeightKind::Trainable => "trainable",
        WeightKind::NonTrainable => "non-trainable",
    }
}

/// Wraps a message as the structural-mismatch error of the load path
#[cold]
fn mismatch(message: String) -> Error {
    Error::Io(IoError::ModelStructureMismatch(message))
}
