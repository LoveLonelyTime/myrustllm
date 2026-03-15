use crate::{
    common::{
        DType, Shape,
        dtype::{DTypeImpl, F32},
        impls::Impl,
    },
    cpu::impls::CPU,
};

pub struct Tensor<I: Impl = CPU, TI: DTypeImpl<I> = F32> {
    pub prototype: TI::Prototype,
}

impl<I: Impl, TI: DTypeImpl<I>> Tensor<I, TI> {
    pub fn new(prototype: TI::Prototype) -> Self {
        Self { prototype }
    }
}

impl<I: Impl, TI: DTypeImpl<I>> TensorPrototype for Tensor<I, TI> {
    fn shape(&self) -> Shape {
        self.prototype.shape()
    }

    fn dtype(&self) -> DType {
        self.prototype.dtype()
    }
}

/// The trait of multi-dimension tensors.
pub trait TensorPrototype {
    /// Return the shape of `&self`.
    fn shape(&self) -> Shape;

    /// Return the dtype of the tensor.
    fn dtype(&self) -> DType;

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

pub type TensorMeta<I: Impl> = (Shape, DType, I::Device);
