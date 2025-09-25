/// This module provides access to the raw data of boston housing dataset
pub mod boston_housing_raw;
/// This module provides access to the raw data of diabetes dataset
pub mod diabetes_raw;
/// This module provides access to the raw data of iris dataset
pub mod iris_raw;
/// This module provides access to the raw data of titanic dataset
pub mod titanic_raw;
/// This module provides access to the raw data of wine quality dataset (Red and White)
pub mod wine_quality_raw;

pub use boston_housing_raw::*;
pub use diabetes_raw::*;
pub use iris_raw::*;
pub use titanic_raw::*;
pub use wine_quality_raw::*;
