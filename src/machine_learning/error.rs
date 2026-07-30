//! Decision-tree error type
//!
//! [`TreeError`] enumerates the failures specific to decision-tree models. Callers receive
//! it through the crate-wide [`Error::Tree`](crate::error::Error::Tree) variant, into which
//! it converts via `?` (a `#[from]` bridge). See [`crate::error`] for the unified
//! [`Error`](crate::error::Error) that aggregates the per-domain error enums

/// Decision-tree-specific errors, surfaced through [`Error::Tree`](crate::error::Error::Tree)
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum TreeError {
    /// A classification-only operation (e.g. `predict_proba`) was called on a regression tree
    #[error("operation requires a classification tree")]
    NotClassificationTree,

    /// The tree's internal structure violated an invariant (a missing child, an absent
    /// categorical fallback, or a leaf without stored probabilities)
    #[error("corrupt tree structure: {0}")]
    CorruptStructure(&'static str),
}

#[cfg(test)]
mod tests {
    use super::TreeError;
    use crate::error::Error;

    /// `#[error(transparent)]` on `Error::Tree` forwards the inner `TreeError`'s Display:
    /// `NotClassificationTree` => `#[error("operation requires a classification tree")]`
    #[test]
    fn display_tree_transparent_forwards_inner() {
        let inner = TreeError::NotClassificationTree;
        assert_eq!(
            inner.to_string(),
            "operation requires a classification tree"
        );
        let outer: Error = Error::from(TreeError::NotClassificationTree);
        assert_eq!(outer.to_string(), inner.to_string());
    }
}
