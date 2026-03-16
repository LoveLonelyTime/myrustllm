use crate::common::dtype::F32;
use crate::common::{DType, DTypeImpl, Impl, Shape};
use crate::cpu::impls::CPU;

/// Tensor is the basic data type in MyRustLLM.
///
/// A Tensor is implemented by a pair(Impl, DTypeImpl<Impl>), where `Impl` is the backend implementation and `DTypeImpl<Impl>` is the data type of the tensor.
pub struct Tensor<I: Impl = CPU, TI: DTypeImpl<I> = F32> {
    pub prototype: TI::Prototype,
}

impl<I: Impl, TI: DTypeImpl<I>> Tensor<I, TI> {
    /// Create a tensor from a prototype.
    ///
    /// If you aren't a lib developer, you should use e.g. alloc, ones, ... to create a tensor instead of `new`.
    pub fn new(prototype: TI::Prototype) -> Self {
        Self { prototype }
    }
}

impl<I: Impl, TI: DTypeImpl<I>> TensorPrototype<I> for Tensor<I, TI> {
    fn shape(&self) -> Shape {
        self.prototype.shape()
    }

    fn dtype(&self) -> DType {
        self.prototype.dtype()
    }

    fn device(&self) -> I::Device {
        self.prototype.device()
    }
}

/// Trait `TensorPrototype` defines three basic functions for tensor prototypes:
/// - shape(): Return the shape of the tensor.
/// - dtype(): Return the dtype of the tensor.
/// - device(): Return the device of the tensor.
///
/// Any tensor prototype should implement `TensorPrototype`.
pub trait TensorPrototype<I: Impl> {
    /// Return the shape of `&self`.
    fn shape(&self) -> Shape;

    /// Return the dtype of the tensor.
    fn dtype(&self) -> DType;

    /// Return the dtype of the tensor.
    fn device(&self) -> I::Device;

    /// Return the dimension of `&self`.
    fn dims(&self) -> usize {
        self.shape().len()
    }

    /// Is `&self` a scalar?
    fn is_scalar(&self) -> bool {
        self.shape().is_scalar()
    }

    /// Return the number of elements.
    fn numel(&self) -> usize {
        self.shape().numel()
    }
}

// TODO: TensorMeta
pub type TensorMeta<I: Impl> = (Shape, DType, I::Device);
