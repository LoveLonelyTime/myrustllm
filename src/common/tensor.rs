use crate::common::{DType, Device, Shape};

/// The trait of multi-dimension tensors.
pub trait Tensor {
    /// Return the shape of `&self`.
    fn shape(&self) -> Shape;

    /// Return the device in which the tensor is accommodating.
    fn device(&self) -> Device;

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
