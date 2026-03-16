//! This mod (view ops) defines operations with view.
//!
//! All operations will not copy data, but just reinterpret the original data.
//!
//! List:
//! - TensorView: view
//! - TensorSlice: slice
//! - TensorBroadcast: broadcast_to

use crate::common::shape::broadcast_shape;
use crate::common::{DTypeImpl, Impl, Shape, Tensor, TensorPrototype};
use std::ops::{Range, RangeFrom, RangeFull, RangeInclusive, RangeTo, RangeToInclusive};

/// Slice type of tensor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorIndex {
    /// i
    Index(isize),
    /// start..end
    Range(isize, isize),
    /// start..
    RangeFrom(isize),
    /// ..end
    RangeTo(isize),
    /// ..
    RangeFull,
    /// start..=end
    RangeInclusive(isize, isize),
    /// ..=end
    RangeToInclusive(isize),
    /// expand a new dim.
    Expand,
    /// ...
    Full,
}

impl From<isize> for TensorIndex {
    fn from(value: isize) -> Self {
        TensorIndex::Index(value)
    }
}

impl From<Range<isize>> for TensorIndex {
    fn from(value: Range<isize>) -> Self {
        TensorIndex::Range(value.start, value.end)
    }
}

impl From<RangeFrom<isize>> for TensorIndex {
    fn from(value: RangeFrom<isize>) -> Self {
        TensorIndex::RangeFrom(value.start)
    }
}

impl From<RangeTo<isize>> for TensorIndex {
    fn from(value: RangeTo<isize>) -> Self {
        TensorIndex::RangeTo(value.end)
    }
}

impl From<RangeFull> for TensorIndex {
    fn from(_: RangeFull) -> Self {
        TensorIndex::RangeFull
    }
}

impl From<RangeInclusive<isize>> for TensorIndex {
    fn from(value: RangeInclusive<isize>) -> Self {
        TensorIndex::RangeInclusive(*value.start(), *value.end())
    }
}

impl From<RangeToInclusive<isize>> for TensorIndex {
    fn from(value: RangeToInclusive<isize>) -> Self {
        TensorIndex::RangeToInclusive(value.end)
    }
}

#[macro_export]
macro_rules! idx {
    ($($slice:expr),* $(,)?) => {
        [ $( idx!(@parse $slice) ),* ]
    };

    (@parse $slice:expr) => {
        TensorIndex::from($slice)
    };
}

/// Tensor view implementation.
pub trait TensorView<I: Impl>: DTypeImpl<I> {
    fn view(src: &Self::Prototype, new_shape: &Shape) -> Option<Self::Prototype>;
}

impl<I: Impl, Src: DTypeImpl<I> + TensorView<I>> Tensor<I, Src> {
    /// Return a new tensor with the same data as the tensor `&self` but of a different shape `new_shape`.
    ///
    /// For a tensor to be viewed, the new view size must be compatible with its original size and stride.
    /// In other words, the new view size must completely merge and split the contiguous subspaces derived from the tensor `&self`.
    /// If `new_shape` is not compatible with its original size and stride, it will return `None`.
    ///
    /// # Exmaples
    ///
    /// ## Contiguous tensors
    ///
    /// ```
    /// use myrustllm::cpu::tensor::CPUTensor;
    /// use myrustllm::cpu::shape::Shape;
    ///
    /// let tensor = CPUTensor::<f32>::from_shape(&Shape::new(vec![4, 5]));
    /// let viewed_tensor = tensor.view(&Shape::new(vec![2, 2, 5])).unwrap();
    /// ```
    ///
    /// ## Auto-inferred dim
    ///
    /// `0` in `new_shape` refers to an auto-inferred dim.
    ///
    /// ```
    /// use myrustllm::cpu::tensor::{Tensor, CPUTensor};
    /// use myrustllm::cpu::shape::Shape;
    ///
    /// let tensor = CPUTensor::<f32>::from_shape(&Shape::new(vec![2, 2, 5]));
    /// let viewed_tensor = tensor.view(&Shape::new(vec![0, 5]));
    ///
    /// assert_eq!(viewed_tensor.shape(), Shape::new(vec![4, 5]));
    /// ```
    pub fn view(&self, new_shape: &Shape) -> Option<Self> {
        Some(Tensor::new(Src::view(&self.prototype, new_shape)?))
    }
}

/// Tensor slice implementation.
pub trait TensorSlice<I: Impl>: DTypeImpl<I> {
    fn slice(src: &Self::Prototype, indices: &[TensorIndex]) -> Self::Prototype;
}

impl<I: Impl, Src: DTypeImpl<I> + TensorSlice<I>> Tensor<I, Src> {
    /// Extract a new slice(view) derived from `&self`.
    ///
    /// The returned CPU tensor will share the same memory with `&self`.
    ///
    /// # Examples
    ///
    /// ```
    /// use myrustllm::cpu::tensor::CPUTensor;
    /// use myrustllm::cpu::slice::TensorIndex;
    /// use myrustllm::idx;
    ///
    /// let tensor = CPUTensor::from_literal([
    ///     [1.0, 2.0],
    ///     [3.0, 4.0]
    /// ]);
    ///
    /// let sliced_tensor = tensor.slice(&idx!(.., 1));
    /// let a = sliced_tensor.slice(&idx!(0));
    /// let b = sliced_tensor.slice(&idx!(-1));
    ///
    /// assert_eq!(a.into_scalar(), 2.0);
    /// assert_eq!(b.into_scalar(), 4.0);
    /// ```
    pub fn slice(&self, indices: &[TensorIndex]) -> Self {
        Tensor::new(Src::slice(&self.prototype, indices))
    }
}

/// Tensor broadcast implementation.
pub trait TensorBroadcast<I: Impl>: DTypeImpl<I> {
    fn broadcast_to(src: &Self::Prototype, target_shape: &Shape) -> Option<Self::Prototype>;
}

impl<I: Impl, Src: DTypeImpl<I> + TensorBroadcast<I>> Tensor<I, Src> {
    /// Broadcast `&self` to the shape `target_shape`.
    ///
    /// Broadcasting doesn't mean allocating new memory, but only creates a new view on the existing tensor.
    /// If the shape `target_shape` is not compatible with the shape of its original shape, it will return `None`.
    ///
    /// # Examples
    ///
    /// ```
    /// use myrustllm::cpu::tensor::CPUTensor;
    /// use myrustllm::cpu::shape::Shape;
    ///
    /// // 1 -> 5
    /// let tensor = CPUTensor::<f32>::from_shape(&Shape::new(vec![2,1,2,3]));
    /// let broadcast_tensor = tensor.broadcast_to(&Shape::new(vec![2,5,2,3])).unwrap();
    /// ```
    pub fn broadcast_to(&self, target_shape: &Shape) -> Option<Self> {
        Some(Tensor::new(Src::broadcast_to(
            &self.prototype,
            target_shape,
        )?))
    }
}

/// Try broadcast two tensor prototypes mutually.
pub fn broadcast_prot<
    I: Impl,
    Lhs: DTypeImpl<I> + TensorBroadcast<I>,
    Rhs: DTypeImpl<I> + TensorBroadcast<I>,
>(
    lhs: &Lhs::Prototype,
    rhs: &Rhs::Prototype,
) -> Option<(Lhs::Prototype, Rhs::Prototype)> {
    let target_shape: Shape = broadcast_shape(&lhs.shape(), &rhs.shape())?;
    Some((
        Lhs::broadcast_to(lhs, &target_shape)?,
        Rhs::broadcast_to(rhs, &target_shape)?,
    ))
}

/// Try broadcast two tensors mutually.
pub fn broadcast<
    I: Impl,
    Lhs: DTypeImpl<I> + TensorBroadcast<I>,
    Rhs: DTypeImpl<I> + TensorBroadcast<I>,
>(
    lhs: &Tensor<I, Lhs>,
    rhs: &Tensor<I, Rhs>,
) -> Option<(Tensor<I, Lhs>, Tensor<I, Rhs>)> {
    let (lhs, rhs) = broadcast_prot::<I, Lhs, Rhs>(&lhs.prototype, &rhs.prototype)?;
    Some((Tensor::new(lhs), Tensor::new(rhs)))
}
