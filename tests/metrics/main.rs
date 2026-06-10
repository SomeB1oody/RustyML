//! Integration tests for the `metrics` feature, with one submodule per metric
//! domain (classification, clustering, regression) and shared helpers in [`common`]

mod common;

mod classification;
mod clustering;
mod regression;
