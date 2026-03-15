//! This mod (cast ops) defines operations with cast & copy.
//!
//! All operations may copy data.
//!
//! List:
//! - TensorCastImpl: cast
//! - TensorCopyImpl: copy
//! - TensorReshapeImpl: reshape

use crate::common::{DTypeImpl, Impl, Shape, Tensor};

/// Tensor cast implement.
pub trait TensorCastImpl<I: Impl, Dst: DTypeImpl<I>>: DTypeImpl<I> {
    fn cast(src: &Self::Prototype) -> Dst::Prototype;
}

impl<I: Impl, Src: DTypeImpl<I>> Tensor<I, Src> {
    /// Return a tensor with type `Dst` from the tensor with type `Src`.
    ///
    /// If `Dst` is not equal to `Src`, it may create a new tensor.
    pub fn cast<Dst: DTypeImpl<I>>(&self) -> Tensor<I, Dst>
    where
        Src: TensorCastImpl<I, Dst>,
    {
        Tensor::new(Src::cast(&self.prototype))
    }
}

/// Tensor copy implement.
pub trait TensorCopyImpl<I: Impl, Src: DTypeImpl<I>>: DTypeImpl<I> {
    fn copy(dst: &mut Self::Prototype, src: &Src::Prototype);
}

impl<I: Impl, Dst: DTypeImpl<I>> Tensor<I, Dst> {
    /// Copy data from the tensor `src`.
    pub fn copy<Src: DTypeImpl<I>>(&mut self, src: &Src::Prototype)
    where
        Dst: TensorCopyImpl<I, Src>,
    {
        Dst::copy(&mut self.prototype, src);
    }
}

/// Tensor reshape implement.
pub trait TensorReshapeImpl<I: Impl>: DTypeImpl<I> {
    fn reshape(src: &Self::Prototype, new_shape: &Shape) -> Self::Prototype;
}

impl<I: Impl, Src: DTypeImpl<I> + TensorReshapeImpl<I>> Tensor<I, Src> {
    /// Return a tensor with the same data copyed from its original data, but with the specified shape `new_shape`.
    ///
    /// If it can be viewed, it may not create a new tensor.
    /// The number of elements must be equal.
    pub fn reshape(&self, new_shape: &Shape) -> Self {
        Tensor::new(Src::reshape(&self.prototype, new_shape))
    }
}
