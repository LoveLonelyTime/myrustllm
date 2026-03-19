//! This mod (reduce ops) defines reduce operations.
//!
//! List:
//! - TensorAddReduce: reduce_add

use crate::common::{DTypeImpl, Impl, Tensor};

/// Tensor add reduce implementation.
pub trait TensorAddReduce<I: Impl>: DTypeImpl<I> {
    fn add_reduce(
        tensor: &Self::Prototype,
        dims: Option<&[usize]>,
        keep_dim: bool,
    ) -> Self::Prototype;
}

impl<I: Impl, TI: DTypeImpl<I> + TensorAddReduce<I>> Tensor<I, TI> {
    pub fn add_reduce(&self, dims: Option<&[usize]>, keep_dim: bool) -> Self {
        Tensor::new(TI::add_reduce(&self.prototype, dims, keep_dim))
    }
}
