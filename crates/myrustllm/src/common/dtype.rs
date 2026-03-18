//! MyRustLLM has two type system:
//! - Dynamic types: These types are defined by DType (u8). Some dispatching system can identify dynamic types from DType ID.
//! - Generic param types: These types are defined by generic params, which are used in rust-lang.

use crate::common::{Impl, TensorPrototype};

/// DType ID (u8).
pub type DType = u8;
pub const DTYPE_F32: DType = 0;
pub const DTYPE_F64: DType = 1;
pub const DTYPE_I32: DType = 2;
pub const DTYPE_I64: DType = 3;

/// Any type implements trait `DTypeOf` can derive a dynamic type.
pub trait DTypeOf {
    const DTYPE: DType;
}

// Generic param types

#[derive(Debug, Clone, Copy)]
pub struct F32;

impl DTypeOf for F32 {
    const DTYPE: DType = DTYPE_F32;
}

#[derive(Debug, Clone, Copy)]
pub struct F64;

impl DTypeOf for F64 {
    const DTYPE: DType = DTYPE_F64;
}

#[derive(Debug, Clone, Copy)]
pub struct I32;

impl DTypeOf for I32 {
    const DTYPE: DType = DTYPE_I32;
}

#[derive(Debug, Clone, Copy)]
pub struct I64;

impl DTypeOf for I64 {
    const DTYPE: DType = DTYPE_I64;
}

/// Trait `DTypeImpl` associate a pair (Impl, Generic param type) with a tensor prototype.
pub trait DTypeImpl<I: Impl> {
    type Prototype: TensorPrototype<I>;
}
