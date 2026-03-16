//! This mod (binary ops) defines binary operations.
//!
//! List:
//! - TensorAdd: +
//! - TensorSub: -
//! - TensorMul: *
//! - TensorDiv: /

use crate::common::{DTypeImpl, Impl, Tensor};

/// Tensor add implementation.
pub trait TensorAdd<I: Impl, Rhs: DTypeImpl<I>>: DTypeImpl<I> {
    type Output: DTypeImpl<I>;
    fn add(
        lhs: &Self::Prototype,
        rhs: &Rhs::Prototype,
    ) -> <Self::Output as DTypeImpl<I>>::Prototype;
}

impl<I: Impl, Lhs: DTypeImpl<I> + TensorAdd<I, Rhs>, Rhs: DTypeImpl<I>>
    std::ops::Add<&Tensor<I, Rhs>> for &Tensor<I, Lhs>
{
    type Output = Tensor<I, Lhs::Output>;
    fn add(self, rhs: &Tensor<I, Rhs>) -> Self::Output {
        Tensor::new(Lhs::add(&self.prototype, &rhs.prototype))
    }
}

/// Tensor sub implementation.
pub trait TensorSub<I: Impl, Rhs: DTypeImpl<I>>: DTypeImpl<I> {
    type Output: DTypeImpl<I>;
    fn sub(
        lhs: &Self::Prototype,
        rhs: &Rhs::Prototype,
    ) -> <Self::Output as DTypeImpl<I>>::Prototype;
}

impl<I: Impl, Lhs: DTypeImpl<I> + TensorSub<I, Rhs>, Rhs: DTypeImpl<I>>
    std::ops::Sub<&Tensor<I, Rhs>> for &Tensor<I, Lhs>
{
    type Output = Tensor<I, Lhs::Output>;
    fn sub(self, rhs: &Tensor<I, Rhs>) -> Self::Output {
        Tensor::new(Lhs::sub(&self.prototype, &rhs.prototype))
    }
}

/// Tensor mul implementation.
pub trait TensorMul<I: Impl, Rhs: DTypeImpl<I>>: DTypeImpl<I> {
    type Output: DTypeImpl<I>;
    fn mul(
        lhs: &Self::Prototype,
        rhs: &Rhs::Prototype,
    ) -> <Self::Output as DTypeImpl<I>>::Prototype;
}

impl<I: Impl, Lhs: DTypeImpl<I> + TensorMul<I, Rhs>, Rhs: DTypeImpl<I>>
    std::ops::Mul<&Tensor<I, Rhs>> for &Tensor<I, Lhs>
{
    type Output = Tensor<I, Lhs::Output>;
    fn mul(self, rhs: &Tensor<I, Rhs>) -> Self::Output {
        Tensor::new(Lhs::mul(&self.prototype, &rhs.prototype))
    }
}

/// Tensor div implementation.
pub trait TensorDiv<I: Impl, Rhs: DTypeImpl<I>>: DTypeImpl<I> {
    type Output: DTypeImpl<I>;
    fn div(
        lhs: &Self::Prototype,
        rhs: &Rhs::Prototype,
    ) -> <Self::Output as DTypeImpl<I>>::Prototype;
}

impl<I: Impl, Lhs: DTypeImpl<I> + TensorDiv<I, Rhs>, Rhs: DTypeImpl<I>>
    std::ops::Div<&Tensor<I, Rhs>> for &Tensor<I, Lhs>
{
    type Output = Tensor<I, Lhs::Output>;
    fn div(self, rhs: &Tensor<I, Rhs>) -> Self::Output {
        Tensor::new(Lhs::div(&self.prototype, &rhs.prototype))
    }
}
