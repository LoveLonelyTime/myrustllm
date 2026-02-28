use std::cell::{Ref, RefCell, RefMut};
use std::iter::zip;
use std::rc::Rc;

use crate::common::{Device, Shape};
use crate::cpu::slice::TensorIndex;
use crate::common::Tensor;
use crate::cuda::interface;
use crate::cuda::mem::{CUDAMemory, CUDAType};

/// Tensor in the CUDA memory
///
/// `T` is the CUDA type
pub struct CUDATensor<T: CUDAType> {
    data: Rc<RefCell<CUDAMemory<T>>>,
    shape: Shape,
    stride: Shape,
    offset: usize,
}

impl<T: CUDAType> Tensor for CUDATensor<T> {
    fn shape(&self) -> Shape {
        self.shape.clone()
    }
    
    fn device(&self) -> crate::common::Device {
        Device::new("cuda", 0)
    }
    
    fn dtype(&self) -> crate::common::DType {
        todo!()
    }
    
}

impl<T: CUDAType> CUDATensor<T> {
    pub fn new(data: CUDAMemory<T>, shape: &Shape, stride: &Shape, offset: usize) -> Self {
        CUDATensor {
            data: Rc::new(RefCell::new(data)),
            shape: shape.clone(),
            stride: stride.clone(),
            offset,
        }
    }

    pub fn from_shape(shape: &Shape) -> Self {
        CUDATensor {
            data: Rc::new(RefCell::new(CUDAMemory::new(shape.numel()))),
            shape: shape.clone(),
            stride: Shape::create_contiguous_stride(shape),
            offset: 0,
        }
    }

    pub fn borrow(&self) -> Ref<'_, CUDAMemory<T>> {
        self.data.borrow()
    }

    pub fn borrow_mut(&mut self) -> RefMut<'_, CUDAMemory<T>> {
        self.data.borrow_mut()
    }

    pub fn stride(&self) -> Shape {
        self.stride.clone()
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn get_memory_stride(&self) -> usize {
        zip(self.shape().iter(), self.stride().iter())
            .fold(1, |acc, (dim, stride)| acc + (dim - 1) * stride)
    }

    pub fn broadcast_to(&self, target_shape: &Shape) -> Option<CUDATensor<T>> {
        if target_shape.len() < self.dims() {
            return None;
        }

        let mut new_stride_v = Vec::with_capacity(target_shape.len());

        let diff = target_shape.len() - self.dims();
        for _ in 0..diff {
            new_stride_v.push(0);
        }

        for i in 0..self.dims() {
            if self.shape()[i] != 1 && self.shape()[i] != target_shape[i + diff] {
                return None;
            }

            if self.shape()[i] == 1 && target_shape[i + diff] != 1 {
                new_stride_v.push(0);
            } else {
                new_stride_v.push(self.stride()[i]);
            }
        }

        Some(CUDATensor {
            data: self.data.clone(),
            shape: target_shape.clone(),
            stride: Shape::new(new_stride_v),
            offset: self.offset,
        })
    }

    /// Copy all values from another CUDA tensor
    pub fn copy_from(&mut self, rhs: &CUDATensor<T>) {
        let broadcast_rhs = rhs.broadcast_to(&self.shape()).expect(&format!(
            "Rhs with shape {} cannot be broadcast to target shape {}!",
            rhs.shape(),
            self.shape()
        ));

        unsafe {
            let rhs_ptr = broadcast_rhs.borrow().as_ptr() as *const libc::c_float;
            let self_ptr = self.borrow_mut().as_mut_ptr() as *mut libc::c_float;

            interface::cuda_tensor_copy_f32(
                rhs_ptr.add(broadcast_rhs.offset()),
                self_ptr.add(self.offset()),
                broadcast_rhs.stride().as_ptr(),
                self.stride().as_ptr(),
                self.shape().as_ptr(),
                self.dims(),
                self.numel(),
            );
        };
    }

    /// Create a new slice(view) derived from `&self`
    ///
    /// The returned CPUTensor shares the same memory with `&self`
    pub fn slice(&self, indices: &[TensorIndex]) -> Self {
        let mut new_shape = Shape::scalar();
        let mut new_stride = Shape::scalar();
        let mut new_offset = self.offset;

        assert!(
            indices.len() <= self.dims(),
            "Too many indices provided. Tensor has {} dimensions but got {} indices.",
            self.dims(),
            indices.len()
        );

        for (dim, index) in indices.iter().enumerate() {
            let dim_size = self.shape[dim];
            let dim_stride = self.stride[dim];

            match index {
                // // Index
                // TensorIndex::Index(i) => {
                //     assert!(
                //         *i < dim_size,
                //         "Index {} out of bounds of dimension {} with size {}.",
                //         i,
                //         dim,
                //         dim_size
                //     );
                //     new_offset += i * dim_stride
                // }

                // // Range
                // TensorIndex::Range(range) => {
                //     assert!(
                //         range.start < dim_size && range.end <= dim_size && range.start < range.end,
                //         "Range {:?} out of bounds of dimension {} with size {}.",
                //         range,
                //         dim,
                //         dim_size
                //     );
                //     new_offset += range.start * dim_stride;
                //     new_shape.push_dim(range.end - range.start);
                //     new_stride.push_dim(dim_stride);
                // }

                // // RangeFrom
                // TensorIndex::RangeFrom(range) => {
                //     assert!(
                //         range.start < dim_size,
                //         "Range {:?} out of bounds of dimension {} with size {}.",
                //         range,
                //         dim,
                //         dim_size
                //     );
                //     new_offset += range.start * dim_stride;
                //     new_shape.push_dim(dim_size - range.start);
                //     new_stride.push_dim(dim_stride);
                // }

                // // RangeTo
                // TensorIndex::RangeTo(range) => {
                //     assert!(
                //         range.end <= dim_size,
                //         "Range {:?} out of bounds of dimension {} with size {}.",
                //         range,
                //         dim,
                //         dim_size
                //     );
                //     new_shape.push_dim(range.end);
                //     new_stride.push_dim(dim_stride);
                // }

                // // RangeFull
                // TensorIndex::RangeFull(_) => {
                //     new_shape.push_dim(dim_size);
                //     new_stride.push_dim(dim_stride);
                // }
                _ => {}
            }
        }

        // Handle remained dimensions
        for dim in indices.len()..self.dims() {
            new_shape.push_dim(self.shape[dim]);
            new_stride.push_dim(self.stride[dim]);
        }

        CUDATensor {
            data: self.data.clone(), // Share
            shape: new_shape,
            stride: new_stride,
            offset: new_offset,
        }
    }
}

pub fn broadcast<T: CUDAType, U: CUDAType>(
    a: &CUDATensor<T>,
    b: &CUDATensor<U>,
) -> Option<(CUDATensor<T>, CUDATensor<U>)> {
    let target_shape = Shape::broadcast_shape(&a.shape(), &b.shape())?;
    Some((
        a.broadcast_to(&target_shape)?,
        b.broadcast_to(&target_shape)?,
    ))
}
