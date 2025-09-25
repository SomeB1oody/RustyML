/// This module provides access to the iris dataset
pub mod iris;

/// This module provides access to the diabetes dataset
pub mod diabetes;

/// This module provides access to the boston housing dataset
pub mod boston_housing;
/// This module provides access to the raw data
pub mod raw_data;
/// This module provides access to the titanic dataset
pub mod titanic;
/// This module provides access to the wine quality dataset (red wine and white wine)
pub mod wine_quality;

pub use boston_housing::*;
pub use diabetes::*;
pub use iris::*;
pub use raw_data::*;
pub use titanic::*;
pub use wine_quality::*;
