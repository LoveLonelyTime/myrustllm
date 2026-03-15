use crate::common::{Impl, TensorPrototype};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum DType {
    F32 = 0,
    F64 = 1,
    I32 = 2,
    I64 = 3,
}

pub trait DTypeOf {
    const DTYPE: DType;
}

impl DTypeOf for f32 {
    const DTYPE: DType = DType::F32;
}

impl DTypeOf for f64 {
    const DTYPE: DType = DType::F64;
}

impl DTypeOf for i32 {
    const DTYPE: DType = DType::I32;
}

impl DTypeOf for i64 {
    const DTYPE: DType = DType::I64;
}

pub trait StdDType: 'static + Copy {
    type RType: Copy;
    const DTYPE: DType;
}

#[derive(Debug, Clone, Copy)]
pub struct F32;

impl StdDType for F32 {
    type RType = f32;
    const DTYPE: DType = DType::F32;
}

#[derive(Debug, Clone, Copy)]
pub struct F64;

impl StdDType for F64 {
    type RType = f64;
    const DTYPE: DType = DType::F64;
}

#[derive(Debug, Clone, Copy)]
pub struct I32;

impl StdDType for I32 {
    type RType = i32;
    const DTYPE: DType = DType::I32;
}

#[derive(Debug, Clone, Copy)]
pub struct I64;

impl StdDType for I64 {
    type RType = i64;
    const DTYPE: DType = DType::I64;
}

#[derive(Debug, Clone, Copy)]
pub struct Any;

pub trait DTypeImpl<I: Impl>: Sized {
    type Prototype: TensorPrototype;
}
