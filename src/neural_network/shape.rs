//! The shape of a tensor as a layer describes it, with a free axis for an extent that no
//! configuration fixes
//!
//! [`Shape`] is what [`UnaryLayer::compute_output_shape`](crate::neural_network::traits::UnaryLayer::compute_output_shape)
//! takes and returns. A layer maps an input shape to an output shape with the layer
//! configuration alone, so a caller gets the answer before any tensor exists

use crate::error::Error;
use crate::{Deserialize, Serialize};
use std::fmt;

/// Shape of a tensor, with 1 entry per axis
///
/// An entry of `Some(n)` is an axis of `n` positions. An entry of `None` is a free axis: the
/// layer configuration does not fix its extent, and any whole number is acceptable there. The
/// batch axis is free in most layers, because 1 layer serves every batch size. A recurrent
/// layer that returns a sequence leaves its time axis free as well
///
/// # Notes
///
/// [`Display`](std::fmt::Display) prints the shape the way
/// [`Sequential::summary`](crate::neural_network::sequential::Sequential::summary) shows it: a
/// parenthesized list, with a free axis printed as `None`. A rank-1 shape whose only axis is
/// free keeps a trailing comma, and a rank-1 shape with a fixed extent does not. The 2 forms
/// come from the 2 renderers that this type replaced
///
/// # Examples
///
/// ```rust
/// use rustyml::neural_network::Shape;
///
/// // A batch of 3 feature vectors of 4 elements each
/// assert_eq!(Shape::known(&[3, 4]).to_string(), "(3, 4)");
///
/// // The same layer, described for every batch size
/// assert_eq!(Shape::with_free_batch(&[3, 4]).to_string(), "(None, 4)");
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct Shape(Vec<Option<usize>>);

impl Shape {
    /// Builds a shape from 1 entry per axis
    ///
    /// # Parameters
    ///
    /// - `axes` - `Some(n)` for an axis of `n` positions, and `None` for a free axis
    ///
    /// # Returns
    ///
    /// - `Shape` - The shape those axes describe
    pub fn new(axes: Vec<Option<usize>>) -> Self {
        Self(axes)
    }

    /// Builds a shape whose every axis has a fixed extent
    ///
    /// # Parameters
    ///
    /// - `dims` - Extent of every axis, batch axis first
    ///
    /// # Returns
    ///
    /// - `Shape` - The shape with no free axis
    pub fn known(dims: &[usize]) -> Self {
        Self(dims.iter().map(|&extent| Some(extent)).collect())
    }

    /// Builds a shape from a full extent list, and frees the batch axis
    ///
    /// Axis 0 becomes free and every later axis keeps its extent. Pass the shape of a real
    /// tensor to describe what the layer accepts for any batch size
    ///
    /// # Parameters
    ///
    /// - `dims` - Extent of every axis, batch axis first. The extent of axis 0 is discarded
    ///
    /// # Returns
    ///
    /// - `Shape` - The shape with a free batch axis
    pub fn with_free_batch(dims: &[usize]) -> Self {
        let mut axes: Vec<Option<usize>> = dims.iter().map(|&extent| Some(extent)).collect();
        if let Some(batch) = axes.first_mut() {
            *batch = None;
        }
        Self(axes)
    }

    /// Builds a shape from a batch axis and the extents of every later axis
    ///
    /// This is the counterpart of [`split_batch`](Shape::split_batch). It puts the batch axis
    /// back in front of a computed extent list, so a free batch axis stays free
    ///
    /// # Parameters
    ///
    /// - `batch` - The batch axis, `None` when it is free
    /// - `tail` - Extent of every axis after the batch axis
    ///
    /// # Returns
    ///
    /// - `Shape` - The joined shape
    pub fn from_batch(batch: Option<usize>, tail: &[usize]) -> Self {
        let mut axes = Vec::with_capacity(tail.len() + 1);
        axes.push(batch);
        axes.extend(tail.iter().map(|&extent| Some(extent)));
        Self(axes)
    }

    /// The same shape with axis 0 made free
    ///
    /// A layer serves every batch size, so the batch extent is not part of what a layer was
    /// built for. A checkpoint records this form, and a load compares this form, so a model
    /// built for 32 samples and a model built for 1 sample carry the same build shape
    ///
    /// # Returns
    ///
    /// - `Shape` - The shape with a free batch axis. A rank-0 shape comes back unchanged
    pub fn free_batch(&self) -> Self {
        let mut axes = self.0.clone();
        if let Some(batch) = axes.first_mut() {
            *batch = None;
        }
        Self(axes)
    }

    /// Number of axes the shape holds
    ///
    /// # Returns
    ///
    /// - `usize` - The rank
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Every axis of the shape, in order
    ///
    /// # Returns
    ///
    /// - `&[Option<usize>]` - `Some(n)` for a fixed axis, and `None` for a free axis
    pub fn axes(&self) -> &[Option<usize>] {
        &self.0
    }

    /// Extent of every axis, when no axis is free
    ///
    /// # Returns
    ///
    /// - `Option<Vec<usize>>` - The extents, or `None` when at least 1 axis is free
    pub fn dims(&self) -> Option<Vec<usize>> {
        self.0.iter().copied().collect()
    }

    /// Checks that the shape has exactly the given rank
    ///
    /// # Parameters
    ///
    /// - `layer` - Layer name, which the message names
    /// - `rank` - Rank the layer accepts
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok` when the rank matches
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the rank is not `rank`
    pub fn check_rank(&self, layer: &str, rank: usize) -> Result<(), Error> {
        if self.0.len() != rank {
            return Err(Error::invalid_input(format!(
                "{layer} expects an input of rank {rank}, got the shape {self} of rank {}",
                self.0.len()
            )));
        }
        Ok(())
    }

    /// Checks that the shape has at least the given rank
    ///
    /// # Parameters
    ///
    /// - `layer` - Layer name, which the message names
    /// - `min_rank` - Lowest rank the layer accepts
    ///
    /// # Returns
    ///
    /// - `Result<(), Error>` - `Ok` when the rank is high enough
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the rank is below `min_rank`
    pub fn check_min_rank(&self, layer: &str, min_rank: usize) -> Result<(), Error> {
        if self.0.len() < min_rank {
            return Err(Error::invalid_input(format!(
                "{layer} expects an input of rank {min_rank} or more, got the shape {self} of \
                 rank {}",
                self.0.len()
            )));
        }
        Ok(())
    }

    /// Splits the batch axis away from the extents of every later axis
    ///
    /// The batch axis comes back as it is, free or fixed. Every later axis must have a fixed
    /// extent, because the shape algebra of a layer reads those extents
    ///
    /// # Parameters
    ///
    /// - `layer` - Layer name, which the message names
    ///
    /// # Returns
    ///
    /// - `Result<(Option<usize>, Vec<usize>), Error>` - The batch axis and the later extents
    ///
    /// # Errors
    ///
    /// - `Error::InvalidInput` - If the shape has no axis, or if an axis after the batch axis
    ///   is free
    pub fn split_batch(&self, layer: &str) -> Result<(Option<usize>, Vec<usize>), Error> {
        let Some((batch, tail)) = self.0.split_first() else {
            return Err(Error::invalid_input(format!(
                "{layer} expects an input with a batch axis, got a shape of rank 0"
            )));
        };

        let mut extents = Vec::with_capacity(tail.len());
        for (position, axis) in tail.iter().enumerate() {
            match axis {
                Some(extent) => extents.push(*extent),
                None => {
                    return Err(Error::invalid_input(format!(
                        "{layer} needs a fixed extent on axis {}, and the shape {self} leaves \
                         that axis free",
                        position + 1
                    )));
                }
            }
        }

        Ok((*batch, extents))
    }
}

impl fmt::Display for Shape {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("(")?;
        for (position, axis) in self.0.iter().enumerate() {
            if position > 0 {
                formatter.write_str(", ")?;
            }
            match axis {
                Some(extent) => write!(formatter, "{extent}")?,
                None => formatter.write_str("None")?,
            }
        }
        // A rank-1 shape whose only axis is free keeps the trailing comma of a 1-element
        // Python tuple, and a rank-1 shape with a fixed extent does not. The 2 forms are what
        // the hand-written renderers produced
        if self.0.len() == 1 && self.0[0].is_none() {
            formatter.write_str(",")?;
        }
        formatter.write_str(")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A shape with every extent fixed prints as a plain tuple
    #[test]
    fn display_prints_a_fixed_shape() {
        assert_eq!(Shape::known(&[2, 3, 4]).to_string(), "(2, 3, 4)");
        assert_eq!(Shape::known(&[1, 1]).to_string(), "(1, 1)");
        assert_eq!(
            Shape::known(&[2, 3, 3, 3, 2]).to_string(),
            "(2, 3, 3, 3, 2)"
        );
    }

    /// A free batch axis prints as "None"
    #[test]
    fn display_prints_a_free_batch_axis() {
        assert_eq!(
            Shape::with_free_batch(&[2, 3, 4]).to_string(),
            "(None, 3, 4)"
        );
        assert_eq!(Shape::with_free_batch(&[7, 12]).to_string(), "(None, 12)");
    }

    /// A recurrent layer that returns a sequence leaves the time axis free as well
    #[test]
    fn display_prints_two_free_axes() {
        let shape = Shape::new(vec![None, None, Some(3)]);
        assert_eq!(shape.to_string(), "(None, None, 3)");
    }

    /// A rank-1 fixed shape carries no trailing comma, and a rank-1 free shape carries one
    ///
    /// The 2 forms come from 2 renderers that this type replaced. This test holds both, and
    /// the `Identity` and `Reshape` tests hold `(None,)` as well
    #[test]
    fn display_keeps_both_rank_1_forms() {
        assert_eq!(Shape::known(&[4]).to_string(), "(4)");
        assert_eq!(Shape::new(vec![None]).to_string(), "(None,)");
    }

    /// A rank-0 shape prints as an empty tuple
    #[test]
    fn display_prints_a_rank_0_shape() {
        assert_eq!(Shape::new(Vec::new()).to_string(), "()");
    }

    /// `dims` reports every extent only when no axis is free
    #[test]
    fn dims_needs_every_axis_fixed() {
        assert_eq!(Shape::known(&[2, 3]).dims(), Some(vec![2, 3]));
        assert_eq!(Shape::with_free_batch(&[2, 3]).dims(), None);
    }

    /// `with_free_batch` frees axis 0 and keeps every later extent
    #[test]
    fn with_free_batch_frees_only_axis_0() {
        let shape = Shape::with_free_batch(&[9, 3, 4]);
        assert_eq!(shape.axes(), &[None, Some(3), Some(4)]);
        assert_eq!(shape.rank(), 3);
    }

    /// `free_batch` frees axis 0 and leaves every other axis alone
    #[test]
    fn free_batch_frees_only_axis_0() {
        let shape = Shape::known(&[9, 3, 4]);
        assert_eq!(shape.free_batch().axes(), &[None, Some(3), Some(4)]);
        assert_eq!(Shape::new(Vec::new()).free_batch(), Shape::new(Vec::new()));
    }

    /// `with_free_batch` on an empty extent list gives an empty shape
    #[test]
    fn with_free_batch_accepts_an_empty_list() {
        assert_eq!(Shape::with_free_batch(&[]), Shape::new(Vec::new()));
    }

    /// `from_batch` puts a batch axis back in front of a computed extent list
    #[test]
    fn from_batch_rejoins_the_batch_axis() {
        assert_eq!(
            Shape::from_batch(None, &[3, 4]),
            Shape::new(vec![None, Some(3), Some(4)])
        );
        assert_eq!(Shape::from_batch(Some(2), &[3]), Shape::known(&[2, 3]));
    }

    /// `check_rank` names the layer, the wanted rank, and the shape it got
    #[test]
    fn check_rank_reports_the_rank_it_wanted() {
        let shape = Shape::with_free_batch(&[2, 3]);
        assert!(shape.check_rank("Conv2D", 2).is_ok());

        let message = shape.check_rank("Conv2D", 4).unwrap_err().to_string();
        assert!(message.contains("Conv2D"), "{message}");
        assert!(message.contains("rank 4"), "{message}");
        assert!(message.contains("(None, 3)"), "{message}");
    }

    /// `check_min_rank` accepts every rank at or above the lowest one
    #[test]
    fn check_min_rank_accepts_a_higher_rank() {
        let shape = Shape::with_free_batch(&[2, 3, 4]);
        assert!(shape.check_min_rank("Dense", 2).is_ok());
        assert!(shape.check_min_rank("Dense", 3).is_ok());

        let message = shape.check_min_rank("Dense", 4).unwrap_err().to_string();
        assert!(message.contains("Dense"), "{message}");
        assert!(message.contains("rank 4 or more"), "{message}");
    }

    /// `split_batch` hands back the batch axis and the later extents
    #[test]
    fn split_batch_keeps_the_batch_axis() {
        let (batch, tail) = Shape::with_free_batch(&[2, 3, 4])
            .split_batch("Dense")
            .unwrap();
        assert_eq!(batch, None);
        assert_eq!(tail, vec![3, 4]);

        let (batch, tail) = Shape::known(&[2, 3]).split_batch("Dense").unwrap();
        assert_eq!(batch, Some(2));
        assert_eq!(tail, vec![3]);
    }

    /// `split_batch` refuses a free axis after the batch axis, and names that axis
    #[test]
    fn split_batch_refuses_a_free_axis() {
        let shape = Shape::new(vec![None, None, Some(3)]);
        let message = shape.split_batch("Conv1D").unwrap_err().to_string();
        assert!(message.contains("Conv1D"), "{message}");
        assert!(message.contains("axis 1"), "{message}");
    }

    /// `split_batch` refuses a shape that holds no axis at all
    #[test]
    fn split_batch_refuses_a_rank_0_shape() {
        let message = Shape::new(Vec::new())
            .split_batch("Flatten")
            .unwrap_err()
            .to_string();
        assert!(message.contains("batch axis"), "{message}");
    }
}
