use crate::common::dtype::F32;
use crate::common::{DType, DTypeImpl, Impl, Shape};
use crate::cpu::impls::CPU;

/// Tensor is the basic data type in MyRustLLM.
///
/// A Tensor is implemented by a pair(Impl, DTypeImpl<Impl>), where `Impl` is the backend implementation and `DTypeImpl<Impl>` is the data type of the tensor.
#[derive(Debug)]
pub struct Tensor<I: Impl = CPU, TI: DTypeImpl<I> = F32> {
    pub prototype: TI::Prototype,
}

impl<I: Impl, TI: DTypeImpl<I, Prototype: Clone>> Clone for Tensor<I, TI> {
    fn clone(&self) -> Self {
        Self {
            prototype: self.prototype.clone(),
        }
    }
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

impl<I: Impl, TI: DTypeImpl<I, Prototype: std::fmt::Display>> std::fmt::Display for Tensor<I, TI>
where
    I::Device: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor({}, shape={:?}, impl={}/{}, device={:?})",
            self.prototype,
            self.shape(),
            std::any::type_name::<I>(),
            std::any::type_name::<TI>(),
            self.device()
        )
    }
}

/// Tensor metadata is a tuple of (Shape, DType, Device).
pub type TensorMetadata<I> = (Shape, DType, <I as Impl>::Device);
