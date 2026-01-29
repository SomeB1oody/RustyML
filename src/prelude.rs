#[cfg(feature = "dataset")]
pub use crate::dataset::*;
#[cfg(feature = "machine_learning")]
pub use crate::machine_learning::*;
#[cfg(feature = "math")]
pub use crate::math::*;
#[cfg(feature = "metric")]
pub use crate::metric::*;
#[cfg(feature = "neural_network")]
pub use crate::neural_network::*;
#[cfg(all(feature = "utility", not(feature = "machine_learning")))]
pub use crate::utility::linear_discriminant_analysis;
#[cfg(feature = "utility")]
pub use crate::utility::{
    kernel_pca, principal_component_analysis, standardize, t_sne, train_test_split,
};
