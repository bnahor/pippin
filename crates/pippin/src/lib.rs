//! Pippin: robot simulation for Apple Silicon.
//!
//! This crate is the CPU reference engine. Its data layout mirrors the Metal
//! backend so both produce the same results.

pub mod batch;
pub mod collision;
pub mod data;
pub mod forward;
pub mod math;
pub mod mesh;
pub mod mjcf;
pub mod model;
pub mod newton;
pub mod simd;
pub mod solver;
pub mod urdf;
pub mod threads;

pub use batch::Batch;
pub use data::Data;
pub use forward::{forward, step};
pub use model::Model;
