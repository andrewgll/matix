#![feature(portable_simd)]

#[cfg(feature="f32")]
pub type FloatType = f32;

#[cfg(feature="f64")]
pub type FloatType = f64;


pub mod matrix;
pub mod matrix_memory_pool;