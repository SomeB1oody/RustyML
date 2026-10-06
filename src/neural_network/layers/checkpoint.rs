//! The named checkpoint format for a model
//!
//! A checkpoint addresses every array of a model by a dotted path. The path is
//! `<layer path>.<name>`: the [`LayerPath`] of the layer that holds the array, and the name that
//! the layer gives the array. A layer at model position 2 gives the path `2.kernel`. A sublayer
//! `forward` of that layer gives the path `2.forward.kernel`. It is the same pair that
//! [`ParamId`] uses for optimizer state, so 1 address
//! serves the training loop and the file alike
//!
//! # What a file holds
//!
//! ```text
//! ModelCheckpoint      magic, format_version, layers
//! LayerCheckpoint      layer_type, build, weights, sublayers
//! SublayerCheckpoint   name, layer
//! WeightRecord         name, kind, shape, data
//! ```
//!
//! A [`LayerCheckpoint`] holds the arrays of 1 layer, and 1 [`SublayerCheckpoint`] per
//! sublayer of that layer. The file therefore holds the same tree as the model
//!
//! The layer type is the string that [`LayerBase::layer_type`] returns, and a load compares it
//! for every node of the tree. Name and shape alone are too weak for that comparison.
//! `InstanceNormalization`, `GroupNormalization`, and a rank-2 `LayerNormalization` all hold
//! `gamma` and `beta` of the same extent. Only the type name tells them apart
//!
//! The kind of a record says whether an optimizer updates the array. It separates the
//! `moving_mean` and the `moving_variance` of [`BatchNormalization`] from the trainable
//! `gamma` and `beta` that stand next to them in the same layer
//!
//! # Strict is the default
//!
//! [`apply`] refuses a file that disagrees with the model in any way, and the refusal names
//! the parameter path. [`apply_partial`] applies what matches and reports the rest, and a
//! caller asks for it by name. A lenient default would load a file that is wrong for the model
//! and leave the difference to show up as a wrong prediction
//!
//! [`LayerBase::layer_type`]: crate::neural_network::traits::LayerBase::layer_type
//! [`LayerPath`]: crate::neural_network::LayerPath
//! [`LayerCheckpoint`]: crate::neural_network::layers::checkpoint::LayerCheckpoint
//! [`SublayerCheckpoint`]: crate::neural_network::layers::checkpoint::SublayerCheckpoint
//! [`apply`]: crate::neural_network::layers::checkpoint::apply
//! [`apply_partial`]: crate::neural_network::layers::checkpoint::apply_partial
//! [`BatchNormalization`]: crate::neural_network::layers::BatchNormalization
//! [`ParamId`]: crate::neural_network::traits::ParamId

use crate::error::{Error, IoError};
use crate::neural_network::Shape;
use crate::neural_network::layer_path::{LayerPath, walk, weight_path};
use crate::neural_network::traits::{Layer, LayerBase, WeightKind};
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
/// Version 4 records the tree of layers at each model position. Each layer record holds the
/// records of its sublayers, and it addresses every array by `<layer path>.<name>`. Version 3
/// held 1 flat record per model position and addressed every array by `<scope>.<name>`.
/// Version 2 recorded a single build shape. Version 1 held a closed enum of per-layer weight
/// containers, and it wrote the variant index of each container in place of any name. No byte
/// of these layouts agrees between versions, so a file from an earlier version stops loading,
/// and the refusal names both versions
///
/// Bump this on any change to a record layout, a field order, or a field meaning. The load
/// path checks the number of model positions, the layer type of each node, the sublayer names
/// of each node, and the name, the kind, and the shape of every array. Those checks can all
/// pass for a file that another release wrote. This number is what makes such a file fail
/// instead of loading values that mean something else
pub const MODEL_FORMAT_VERSION: u32 = 4;

/// The shapes that a layer was built for
///
/// A layer allocates every array it owns in
/// [`Layer::build_many`], from the shapes of
/// its inputs. The build shape is therefore the 1 thing a fresh layer does not have, and it
/// decides every extent the layer allocates. A file carries it so that a load can refuse a
/// model that was built for another input
///
/// The record holds 1 shape per input of the layer. Almost every layer takes 1 input, and a
/// merge layer takes several
///
/// The batch axis is free. A layer serves every batch size, so the batch extent is not part of
/// what the layer was built for. A model built for 32 samples takes the checkpoint of a model
/// built for 1
///
/// A load compares the field when the file and the layer both carry one, and skips the
/// comparison in every other case. A layer that owns no array and reads no extent of its
/// input, such as an activation, carries none
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BuildConfig {
    /// Shape of every input the layer was built for, each with a first and free batch axis
    pub input_shapes: Vec<Shape>,
}

impl BuildConfig {
    /// The build record of a layer that built for `inputs`
    ///
    /// The batch axis of every shape is freed here, so every caller records the same canonical
    /// form
    ///
    /// # Parameters
    ///
    /// - `inputs` - 1 shape per input of the layer, batch axis first
    ///
    /// # Returns
    ///
    /// - `BuildConfig` - The record, with a free batch axis on every shape
    #[inline]
    pub fn new(inputs: &[Shape]) -> Self {
        Self {
            input_shapes: inputs.iter().map(Shape::free_batch).collect(),
        }
    }

    /// The build record of a layer with 1 input that built for `input`
    ///
    /// # Parameters
    ///
    /// - `input` - Shape the layer built for, batch axis first
    ///
    /// # Returns
    ///
    /// - `BuildConfig` - The record, holding that 1 shape with a free batch axis
    #[inline]
    pub fn unary(input: &Shape) -> Self {
        Self {
            input_shapes: vec![input.free_batch()],
        }
    }

    /// The shapes as 1 line, for an error message
    ///
    /// # Returns
    ///
    /// - `String` - The shapes, separated by a comma when the layer takes several inputs
    fn describe(&self) -> String {
        self.input_shapes
            .iter()
            .map(Shape::to_string)
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// 1 named array of 1 layer, as a file holds it
///
/// The `'a` lifetime is threaded from the layer: saving borrows the live array and the live
/// name, and loading owns both
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightRecord<'a> {
    /// Name the layer gives the array. It is the last part of the checkpoint path
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
/// The record holds the arrays of the layer itself, and 1 [`SublayerCheckpoint`] per sublayer
/// of the layer. A layer that holds no sublayer gives an empty list
///
/// The `'a` lifetime is threaded from the layer, as in [`WeightRecord`]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerCheckpoint<'a> {
    /// The string that [`LayerBase::layer_type`] returned. A load compares it against the layer
    /// at the same path
    ///
    /// [`LayerBase::layer_type`]: crate::neural_network::traits::LayerBase::layer_type
    pub layer_type: Cow<'a, str>,
    /// The shape the layer was built for, when the layer reports one. See [`BuildConfig`]
    pub build: Option<BuildConfig>,
    /// Every array of the layer itself, in the order [`LayerBase::weights`] gives them
    ///
    /// [`LayerBase::weights`]: crate::neural_network::traits::LayerBase::weights
    pub weights: Vec<WeightRecord<'a>>,
    /// Every sublayer of the layer, in the order [`LayerBase::sublayers`] gives them
    ///
    /// [`LayerBase::sublayers`]: crate::neural_network::traits::LayerBase::sublayers
    pub sublayers: Vec<SublayerCheckpoint<'a>>,
}

/// 1 sublayer of a layer, as a file holds it
///
/// The `'a` lifetime is threaded from the layer, as in [`WeightRecord`]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SublayerCheckpoint<'a> {
    /// The name that the holding layer gives the sublayer. It is 1 part of the checkpoint path
    pub name: Cow<'a, str>,
    /// The record of the sublayer
    pub layer: LayerCheckpoint<'a>,
}

/// A whole model, as a file holds it
///
/// The `'a` lifetime is threaded from the layers, as in [`WeightRecord`]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCheckpoint<'a> {
    /// Magic tag identifying a RustyML model file. See [`MODEL_MAGIC`]
    ///
    /// A file saved before this tag existed is refused rather than loaded with wrong values
    pub magic: u32,
    /// On-disk model format version of this file. See [`MODEL_FORMAT_VERSION`]
    pub format_version: u32,
    /// Every model position, from the input
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

/// Reads every array of every layer and every sublayer into a checkpoint that borrows the
/// model
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
    ModelCheckpoint {
        magic: MODEL_MAGIC,
        format_version: MODEL_FORMAT_VERSION,
        layers: layers.iter().map(|layer| capture_layer(&**layer)).collect(),
    }
}

/// Reads 1 layer and all of its sublayers into a record that borrows the layer
fn capture_layer(layer: &dyn LayerBase) -> LayerCheckpoint<'_> {
    LayerCheckpoint {
        layer_type: Cow::Borrowed(layer.layer_type()),
        build: layer.build_config(),
        weights: layer
            .weights()
            .into_iter()
            .map(|entry| WeightRecord {
                name: Cow::Borrowed(entry.name),
                kind: entry.kind,
                shape: entry.value.shape().to_vec(),
                // `to_slice` keeps the lifetime of the layer, so a contiguous array is
                // borrowed here instead of copied
                data: match entry.value.to_slice() {
                    Some(run) => Cow::Borrowed(run),
                    None => Cow::Owned(entry.value.iter().copied().collect()),
                },
            })
            .collect(),
        sublayers: layer
            .sublayers()
            .into_iter()
            .map(|sub| SublayerCheckpoint {
                name: sub.name,
                layer: capture_layer(sub.layer),
            })
            .collect(),
    }
}

/// Applies a checkpoint to a model, and refuses every disagreement
///
/// The load runs in 2 passes, and the first one writes nothing. It checks the number of model
/// positions first. Then it visits every node of every layer tree, and it checks, in this
/// order:
///
/// 1. The layer type of the node.
/// 2. The build shape of the node, when the file and the layer both carry one.
/// 3. The number of arrays of the node, and the name of each one, in order.
/// 4. The kind of each array.
/// 5. The shape of each array, and the element count of the record.
/// 6. The number of sublayers of the node, and the name of each one, in order.
///
/// The second pass writes. A refusal therefore leaves the model exactly as it was. A model
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
///   message names the checkpoint path, or the layer path for a whole-layer disagreement
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
        check_layer(&LayerPath::root(scope), &**layer, saved)?;
    }

    // Pass 2 writes, and every check above already passed
    for (layer, saved) in layers.iter_mut().zip(file.layers.iter()) {
        write_layer(&mut **layer, saved);
    }

    Ok(())
}

/// Compares 1 node and all of its sublayers against the record of the file, and writes nothing
fn check_layer(
    path: &LayerPath,
    layer: &dyn LayerBase,
    saved: &LayerCheckpoint<'_>,
) -> Result<(), Error> {
    let layer_type = layer.layer_type();
    if layer_type != saved.layer_type {
        return Err(mismatch(format!(
            "layer `{path}` type mismatch: model has `{layer_type}`, file has `{}`",
            saved.layer_type
        )));
    }

    // A layer that owns no array carries no build shape, so this compares only when both
    // sides carry one
    if let (Some(wanted), Some(found)) = (layer.build_config(), saved.build.as_ref())
        && wanted != *found
    {
        return Err(mismatch(format!(
            "layer `{path}` (`{layer_type}`) was built for input shape {}, and the file records \
             {}",
            wanted.describe(),
            found.describe()
        )));
    }

    let targets = layer.weights();
    if targets.len() != saved.weights.len() {
        // Name both rosters. An optional parameter such as a bias is exactly the case where
        // the counts differ, and the names say which array is the extra one
        return Err(mismatch(format!(
            "layer `{path}` (`{layer_type}`) holds the arrays {:?}, and the file holds {:?}",
            targets.iter().map(|e| e.name).collect::<Vec<_>>(),
            saved.weights.iter().map(|r| &*r.name).collect::<Vec<_>>()
        )));
    }

    for (target, record) in targets.iter().zip(saved.weights.iter()) {
        let at = weight_path(path, target.name);
        if target.name != record.name {
            return Err(mismatch(format!(
                "the model holds `{at}` where the file holds `{}`",
                weight_path(path, &record.name)
            )));
        }
        if target.kind != record.kind {
            return Err(mismatch(format!(
                "`{at}` is {} in the model, and {} in the file",
                kind_word(target.kind),
                kind_word(record.kind)
            )));
        }
        if target.value.shape() != record.shape.as_slice() {
            return Err(mismatch(format!(
                "`{at}` has shape {:?} in the model, and {:?} in the file",
                target.value.shape(),
                record.shape
            )));
        }
        if record.data.len() != target.value.len() {
            return Err(mismatch(format!(
                "`{at}` holds {} elements at shape {:?}, and the file record carries {}",
                target.value.len(),
                record.shape,
                record.data.len()
            )));
        }
    }

    let subs = layer.sublayers();
    let model_names: Vec<&str> = subs.iter().map(|sub| &*sub.name).collect();
    let file_names: Vec<&str> = saved.sublayers.iter().map(|sub| &*sub.name).collect();
    if model_names != file_names {
        return Err(mismatch(format!(
            "layer `{path}` (`{layer_type}`) holds the sublayers {model_names:?}, and the file \
             holds {file_names:?}"
        )));
    }
    for (sub, record) in subs.iter().zip(saved.sublayers.iter()) {
        check_layer(&path.child(sub.name.clone()), sub.layer, &record.layer)?;
    }
    Ok(())
}

/// Writes the arrays of 1 record into 1 node and all of its sublayers
///
/// [`check_layer`] already compared the 2 trees, so every roster lines up
fn write_layer(layer: &mut dyn LayerBase, saved: &LayerCheckpoint<'_>) {
    for (target, record) in layer.weights_mut().iter_mut().zip(saved.weights.iter()) {
        write_record(&mut target.value, &record.data);
    }
    for (sub, record) in layer
        .sublayers_mut()
        .into_iter()
        .zip(saved.sublayers.iter())
    {
        write_layer(sub.layer, &record.layer);
    }
}

/// Applies what the file and the model agree on, and reports the rest
///
/// This is the opt-in lenient load. It writes an array only when 3 conditions all hold: the
/// node holds the same layer type, the 2 sides agree on the build shape, and the file holds
/// the same name, kind, and shape. Everything else goes into the report and nothing else
/// fails.
///
/// A node whose layer type differs contributes every path of its tree to both lists. A name
/// and a shape cannot tell 2 normalization layers apart
///
/// A node whose build shape differs does the same. [`apply`] refuses such a file, and this
/// skips the node and its sublayers: the 2 paths agree that a layer built for another input
/// takes no weights. The per-array shape check cannot stand in for it, because a convolution
/// kernel is the same shape for every spatial extent
///
/// A sublayer of the model meets the sublayer of the file with the same name. A sublayer that
/// only 1 side holds contributes every path of its tree to 1 list
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
        partial_layer(
            &LayerPath::root(scope),
            &mut **layer,
            file.layers.get(scope),
            &mut report,
        );
    }

    // A file longer than the model has whole layers that reached nothing
    for (scope, saved) in file.layers.iter().enumerate().skip(layers.len()) {
        file_paths(&LayerPath::root(scope), saved, &mut report.unused);
    }

    report
}

/// Applies what 1 record and 1 node agree on, and reports the rest, for the node and all of
/// its sublayers
fn partial_layer(
    path: &LayerPath,
    layer: &mut dyn LayerBase,
    at_path: Option<&LayerCheckpoint<'_>>,
    report: &mut LoadReport,
) {
    let saved = at_path.filter(|saved| saved.layer_type == layer.layer_type());
    let Some(saved) = saved else {
        // The path holds another layer type, or the file holds no record at the path.
        // Neither side reaches the other, so every path of both sides is left over
        model_paths(path, layer, &mut report.missing);
        if let Some(other) = at_path {
            file_paths(path, other, &mut report.unused);
        }
        return;
    };

    // A build shape mismatch skips the whole node. The per-array shape check below cannot
    // catch it, because a kernel shape does not depend on the spatial extent
    if let (Some(wanted), Some(found)) = (layer.build_config(), saved.build.as_ref())
        && wanted != *found
    {
        model_paths(path, layer, &mut report.missing);
        file_paths(path, saved, &mut report.unused);
        return;
    }

    let mut taken = vec![false; saved.weights.len()];
    for target in layer.weights_mut().iter_mut() {
        let at = weight_path(path, target.name);
        // `taken` stops 2 target arrays from matching 1 record
        let found = saved
            .weights
            .iter()
            .enumerate()
            .find(|(index, record)| {
                !taken[*index]
                    && record.name == target.name
                    && record.kind == target.kind
                    && record.shape.as_slice() == target.value.shape()
                    && record.data.len() == target.value.len()
            })
            .map(|(index, _)| index);
        match found {
            Some(index) => {
                taken[index] = true;
                write_record(&mut target.value, &saved.weights[index].data);
                report.applied.push(at);
            }
            None => report.missing.push(at),
        }
    }
    report.unused.extend(
        saved
            .weights
            .iter()
            .zip(taken.iter())
            .filter(|(_, used)| !**used)
            .map(|(record, _)| weight_path(path, &record.name)),
    );

    let mut sub_taken = vec![false; saved.sublayers.len()];
    for sub in layer.sublayers_mut() {
        let sub_path = path.child(sub.name.clone());
        // A model build refuses 2 sublayers of 1 name, and `sub_taken` keeps 2 file records
        // of 1 name from both reaching it
        let found = saved
            .sublayers
            .iter()
            .enumerate()
            .find(|(index, record)| !sub_taken[*index] && record.name == sub.name)
            .map(|(index, _)| index);
        if let Some(index) = found {
            sub_taken[index] = true;
        }
        partial_layer(
            &sub_path,
            sub.layer,
            found.map(|index| &saved.sublayers[index].layer),
            report,
        );
    }
    for (record, used) in saved.sublayers.iter().zip(sub_taken.iter()) {
        if !*used {
            file_paths(
                &path.child(record.name.to_string()),
                &record.layer,
                &mut report.unused,
            );
        }
    }
}

/// Appends the checkpoint path of every array of 1 node and all of its sublayers
fn model_paths(path: &LayerPath, layer: &dyn LayerBase, into: &mut Vec<String>) {
    walk(layer, path, &mut |node_path, node| {
        into.extend(
            node.weights()
                .iter()
                .map(|e| weight_path(node_path, e.name)),
        );
    });
}

/// Appends the checkpoint path of every array of 1 record and all of its sublayer records
fn file_paths(path: &LayerPath, saved: &LayerCheckpoint<'_>, into: &mut Vec<String>) {
    into.extend(saved.weights.iter().map(|r| weight_path(path, &r.name)));
    for sub in &saved.sublayers {
        file_paths(&path.child(sub.name.to_string()), &sub.layer, into);
    }
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
