use std::cell::{Ref, RefCell, RefMut};
use std::iter::zip;
use std::rc::Rc;

use crate::cpu::shape::{Shape, create_contiguous_stride};
use crate::cpu::tensor::Tensor;
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

impl<T: CUDAType> Tensor<T> for CUDATensor<T> {
    fn shape(&self) -> Shape {
        self.shape.clone()
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
            stride: create_contiguous_stride(shape),
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
}

pub fn broadcast_shape<T: CUDAType>(a: &CUDATensor<T>, b: &CUDATensor<T>) -> Option<Shape> {
    let shape_a = a.shape();
    let shape_b = b.shape();

    let max_dims = std::cmp::max(a.dims(), b.dims());

    let mut result_shape_v = Vec::with_capacity(max_dims);

    for i in 0..max_dims {
        let d_a = if i < a.dims() {
            shape_a[a.dims() - 1 - i]
        } else {
            1
        };
        let d_b = if i < b.dims() {
            shape_b[b.dims() - 1 - i]
        } else {
            1
        };

        if d_a != d_b && d_a != 1 && d_b != 1 {
            return None;
        }

        result_shape_v.push(std::cmp::max(d_a, d_b));
    }
    result_shape_v.reverse();

    return Some(Shape::new(result_shape_v));
}

pub fn broadcast<T: CUDAType>(
    a: &CUDATensor<T>,
    b: &CUDATensor<T>,
) -> Option<(CUDATensor<T>, CUDATensor<T>)> {
    let target_shape = broadcast_shape(a, b)?;
    Some((
        a.broadcast_to(&target_shape)?,
        b.broadcast_to(&target_shape)?,
    ))
}
