//! This mod (cast ops) defines operations with cast & copy.
//!
//! All operations may copy data.
//!
//! List:
//! - TensorCast: cast
//! - TensorCopy: copy
//! - TensorReshape: reshape

use crate::common::{DTypeImpl, Impl, Shape, Tensor};

/// Tensor cast implementation.
pub trait TensorCast<I: Impl, Dst: DTypeImpl<I>>: DTypeImpl<I> {
    fn cast(src: &Self::Prototype) -> Dst::Prototype;
}

impl<I: Impl, Src: DTypeImpl<I>> Tensor<I, Src> {
    /// Return a tensor with type `Dst` from the tensor with type `Src`.
    ///
    /// If `Dst` is not equal to `Src`, it may create a new tensor.
    pub fn cast<Dst: DTypeImpl<I>>(&self) -> Tensor<I, Dst>
    where
        Src: TensorCast<I, Dst>,
    {
        Tensor::new(Src::cast(&self.prototype))
    }
}

/// Tensor copy implementation.
pub trait TensorCopy<I: Impl, Src: DTypeImpl<I>>: DTypeImpl<I> {
    fn copy(dst: &mut Self::Prototype, src: &Src::Prototype);
}

impl<I: Impl, Dst: DTypeImpl<I>> Tensor<I, Dst> {
    /// Copy data from the tensor `src`.
    pub fn copy<Src: DTypeImpl<I>>(&mut self, src: &Src::Prototype)
    where
        Dst: TensorCopy<I, Src>,
    {
        Dst::copy(&mut self.prototype, src);
    }
}

/// Tensor reshape implementation.
pub trait TensorReshape<I: Impl>: DTypeImpl<I> {
    fn reshape(src: &Self::Prototype, new_shape: &Shape) -> Self::Prototype;
}

impl<I: Impl, Src: DTypeImpl<I> + TensorReshape<I>> Tensor<I, Src> {
    /// Return a tensor with the same data copyed from its original data, but with the specified shape `new_shape`.
    ///
    /// If it can be viewed, it may not create a new tensor.
    /// The number of elements must be equal.
    pub fn reshape(&self, new_shape: &Shape) -> Self {
        Tensor::new(Src::reshape(&self.prototype, new_shape))
    }
}
