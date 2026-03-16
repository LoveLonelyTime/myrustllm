//! This mod (init) defines how to init a tensor.
//!
//! List:
//! - TensorAllocInit: alloc
//! - TensorZerosInit: zeros
//! - TensorOnesInit: ones
//! - TensorRawDataInit: from_raw.

use crate::common::io::{Literal, TensorRawData};
use crate::common::{DTypeImpl, Impl, Shape, Tensor};

/// Tensor alloc init implementation.
pub trait TensorAllocInit<I: Impl>: DTypeImpl<I> {
    fn init(shape: &Shape, device: &I::Device) -> Self::Prototype;
}

impl<I: Impl, TI: DTypeImpl<I> + TensorAllocInit<I>> Tensor<I, TI> {
    /// Allocate a tensor without any initial data.
    ///
    /// Alloc just allocates a memory for a tensor, which may contain dirty data.
    pub fn alloc(shape: &Shape, device: &I::Device) -> Self {
        Tensor::new(TI::init(shape, device))
    }
}

/// Tensor zeros init implementation.
pub trait TensorZerosInit<I: Impl>: DTypeImpl<I> {
    fn init(shape: &Shape, device: &I::Device) -> Self::Prototype;
}

impl<I: Impl, TI: DTypeImpl<I> + TensorZerosInit<I>> Tensor<I, TI> {
    /// Create a tensor with zero data.
    pub fn zeros(shape: &Shape, device: &I::Device) -> Self {
        Tensor::new(TI::init(shape, device))
    }
}

/// Tensor ones init implementation.
pub trait TensorOnesInit<I: Impl>: DTypeImpl<I> {
    fn init(shape: &Shape, device: &I::Device) -> Self::Prototype;
}

impl<I: Impl, TI: DTypeImpl<I> + TensorOnesInit<I>> Tensor<I, TI> {
    /// Create a tensor with one data.
    pub fn ones(shape: &Shape, device: &I::Device) -> Self {
        Tensor::new(TI::init(shape, device))
    }
}

/// Tensor raw data init implementation.
pub trait TensorRawDataInit<I: Impl>: DTypeImpl<I> {
    fn init(data: impl Into<TensorRawData>, device: &I::Device) -> Self::Prototype;
}

impl<I: Impl, TI: DTypeImpl<I> + TensorRawDataInit<I>> Tensor<I, TI> {
    /// Create a tensor from raw data.
    pub fn from_raw(data: impl Into<TensorRawData>, device: &I::Device) -> Self {
        Tensor::new(TI::init(data, device))
    }

    /// Create a tensor from literal.
    pub fn from_literal(literal: impl Literal, device: &I::Device) -> Self {
        Self::from_raw(literal, device)
    }
}
