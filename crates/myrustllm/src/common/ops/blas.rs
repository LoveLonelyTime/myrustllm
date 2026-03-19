//! This mod (blas) defines BLAS(Basic Linear Algebra Subprograms) operations.
//!
//! List:
//! - TensorMatmul: matmul

use crate::common::{DTypeImpl, Impl, Tensor};

/// Tensor matmul implementation.
pub trait TensorMatmul<I: Impl, Rhs: DTypeImpl<I>>: DTypeImpl<I> {
    type Output: DTypeImpl<I>;
    fn matmul(
        lhs: &Self::Prototype,
        rhs: &Rhs::Prototype,
    ) -> <Self::Output as DTypeImpl<I>>::Prototype;
}

impl<I: Impl, Lhs: DTypeImpl<I>> Tensor<I, Lhs> {
    pub fn matmul<Rhs>(&self, rhs: &Tensor<I, Rhs>) -> Tensor<I, Lhs::Output>
    where
        Lhs: TensorMatmul<I, Rhs>,
        Rhs: DTypeImpl<I>,
    {
        Tensor::new(Lhs::matmul(&self.prototype, &rhs.prototype))
    }
}
