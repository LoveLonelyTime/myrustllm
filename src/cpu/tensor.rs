use std::cell::{Ref, RefCell, RefMut};
use std::env;
use std::iter::zip;
use std::rc::Rc;

use crate::cpu::mem::CPUMemory;
use crate::cpu::shape::{Shape, broadcast_shape, create_contiguous_stride};
use crate::cpu::slice::TensorSlice;

/// Flattenable data
pub trait Flattenable<T> {
    /// Get the original shape
    fn shape(&self) -> Shape;

    /// Flatten the data and append to `out`
    fn flatten(&self, out: &mut Vec<T>);
}

/// The implementation for array
impl<T, U: Flattenable<T>, const N: usize> Flattenable<T> for [U; N] {
    fn shape(&self) -> Shape {
        let mut shape = Shape::new(vec![N]);
        if N > 0 {
            shape.extend_dim(&self[0].shape());
        }
        shape
    }

    fn flatten(&self, out: &mut Vec<T>) {
        for item in self {
            item.flatten(out);
        }
    }
}

macro_rules! flattenable_scalar {
    ($t:ty) => {
        impl Flattenable<$t> for $t {
            fn shape(&self) -> Shape {
                Shape::scalar()
            }

            fn flatten(&self, out: &mut Vec<$t>) {
                out.push(*self);
            }
        }
    };
}

flattenable_scalar!(f32);
flattenable_scalar!(f64);
flattenable_scalar!(i32);
flattenable_scalar!(i64);
flattenable_scalar!(u32);
flattenable_scalar!(u8);

/// Multi-dimension tensor
pub trait Tensor {
    /// Return the shape of `&self`
    fn shape(&self) -> Shape;

    /// Return the dimension of `&self`
    fn dims(&self) -> usize {
        self.shape().len()
    }

    /// Is `&self` a scalar?
    fn is_scalar(&self) -> bool {
        self.shape().is_scalar()
    }

    // Return the number of elements
    fn numel(&self) -> usize {
        self.shape().numel()
    }
}

/// Tensor in the CPU memory
///
pub struct CPUTensor<T: Copy> {
    data: Rc<RefCell<CPUMemory<T>>>,
    shape: Shape,
    stride: Shape,
    offset: usize,
}

impl<T: Copy> Tensor for CPUTensor<T> {
    fn shape(&self) -> Shape {
        self.shape.clone()
    }
}

impl<T: Copy> CPUTensor<T> {
    /// Create a new CPU tensor
    pub fn new(data: CPUMemory<T>, shape: &Shape, stride: &Shape, offset: usize) -> Self {
        CPUTensor {
            data: Rc::new(RefCell::new(data)),
            shape: shape.clone(),
            stride: stride.clone(),
            offset,
        }
    }

    /// Create a new CPU tensor with shape `shape`
    pub fn from_shape(shape: &Shape) -> Self {
        CPUTensor {
            data: Rc::new(RefCell::new(CPUMemory::new(shape.numel()))),
            shape: shape.clone(),
            stride: create_contiguous_stride(shape),
            offset: 0,
        }
    }

    /// Create a new CPU tensor filled with `input`
    pub fn fill(shape: &Shape, input: T) -> Self {
        let mut mem = CPUMemory::<T>::new(shape.numel());
        unsafe {
            for i in 0..mem.size() {
                mem.as_mut_ptr().add(i).write(input.into());
            }
        };

        CPUTensor {
            data: Rc::new(RefCell::new(mem)),
            shape: shape.clone(),
            stride: create_contiguous_stride(shape),
            offset: 0,
        }
    }

    /// Create a new CPU tensor from the array `input`
    ///
    /// # Example
    ///
    /// ```
    /// use myrustllm::cpu::tensor::CPUTensor;
    ///
    /// let tensor = CPUTensor::<f32>::from_array([
    ///     [1.0, 1.0],
    ///     [2.0, 2.0]
    /// ]);
    /// ```
    pub fn from_array<U: Flattenable<T>>(input: U) -> Self {
        let mut data = Vec::new();
        input.flatten(&mut data);
        let shape = input.shape();
        let stride = create_contiguous_stride(&shape);

        let mut mem = CPUMemory::<T>::new(shape.numel());

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), mem.as_mut_ptr(), data.len());
        };

        CPUTensor {
            data: Rc::new(RefCell::new(mem)),
            shape,
            stride,
            offset: 0,
        }
    }

    /// Create a new CPU scalar
    pub fn scalar(input: T) -> Self {
        let shape = Shape::scalar();
        let stride = create_contiguous_stride(&shape);
        let mut mem = CPUMemory::<T>::new(1);
        unsafe {
            *mem.as_mut_ptr() = input;
        };

        CPUTensor {
            data: Rc::new(RefCell::new(mem)),
            shape,
            stride,
            offset: 0,
        }
    }

    /// Borrow a ref
    pub fn borrow(&self) -> Ref<'_, CPUMemory<T>> {
        self.data.borrow()
    }

    /// Borrow a mutable ref
    pub fn borrow_mut(&mut self) -> RefMut<'_, CPUMemory<T>> {
        self.data.borrow_mut()
    }

    /// Return the stride of the tensor
    pub fn stride(&self) -> Shape {
        self.stride.clone()
    }

    /// Return the offset of the tensor
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Return the total stride of the tensor
    pub fn get_memory_stride(&self) -> usize {
        zip(self.shape().iter(), self.stride().iter())
            .fold(1, |acc, (dim, stride)| acc + (dim - 1) * stride)
    }

    /// Return the ref of the element at `indices`
    ///
    /// # Example
    ///
    /// ```
    /// use myrustllm::cpu::tensor::CPUTensor;
    ///
    /// let tensor = CPUTensor::<f32>::from_array([
    ///     [1.0, 3.0],
    ///     [2.0, 2.0]
    /// ]);
    ///
    /// assert_eq!(*tensor.index(&[0, 1]), 3.0);
    /// ```
    pub fn index(&self, indices: &[usize]) -> Ref<'_, T> {
        assert!(
            indices.len() == self.dims(),
            "Indices with size {} don't match the shape with size {}.",
            indices.len(),
            self.dims()
        );

        let mut index = self.offset;
        for (dim, i) in indices.iter().enumerate() {
            assert!(
                *i < self.shape()[dim],
                "Index {} of dim {} out of bounds of the shape with size {}.",
                i,
                dim,
                self.shape()[dim]
            );
            index += i * self.stride[dim];
        }
        Ref::map(self.data.borrow(), |v| unsafe {
            v.as_ptr().add(index).as_ref().unwrap()
        })
    }

    /// Return the mutable ref of the element at `indices`
    pub fn index_mut(&mut self, indices: &[usize]) -> RefMut<'_, T> {
        assert!(
            indices.len() == self.dims(),
            "Indices with size {} don't match the shape with size {}.",
            indices.len(),
            self.dims()
        );

        let mut index = self.offset;
        for (dim, i) in indices.iter().enumerate() {
            assert!(
                *i < self.shape()[dim],
                "Index {} of dim {} out of bounds of the shape with size {}.",
                i,
                dim,
                self.shape()[dim]
            );
            index += i * self.stride[dim];
        }
        RefMut::map(self.data.borrow_mut(), |v| unsafe {
            v.as_mut_ptr().add(index).as_mut().unwrap()
        })
    }

    /// Create a new slice(view) derived from `&self`
    ///
    /// The returned CPUTensor shares the same memory with `&self`
    pub fn slice(&self, indices: &[TensorSlice]) -> Self {
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
                // Index
                TensorSlice::Index(i) => {
                    assert!(
                        *i < dim_size,
                        "Index {} out of bounds of dimension {} with size {}.",
                        i,
                        dim,
                        dim_size
                    );
                    new_offset += i * dim_stride
                }

                // Range
                TensorSlice::Range(range) => {
                    assert!(
                        range.start < dim_size && range.end <= dim_size && range.start < range.end,
                        "Range {:?} out of bounds of dimension {} with size {}.",
                        range,
                        dim,
                        dim_size
                    );
                    new_offset += range.start * dim_stride;
                    new_shape.push_dim(range.end - range.start);
                    new_stride.push_dim(dim_stride);
                }

                // RangeFrom
                TensorSlice::RangeFrom(range) => {
                    assert!(
                        range.start < dim_size,
                        "Range {:?} out of bounds of dimension {} with size {}.",
                        range,
                        dim,
                        dim_size
                    );
                    new_offset += range.start * dim_stride;
                    new_shape.push_dim(dim_size - range.start);
                    new_stride.push_dim(dim_stride);
                }

                // RangeTo
                TensorSlice::RangeTo(range) => {
                    assert!(
                        range.end <= dim_size,
                        "Range {:?} out of bounds of dimension {} with size {}.",
                        range,
                        dim,
                        dim_size
                    );
                    new_shape.push_dim(range.end);
                    new_stride.push_dim(dim_stride);
                }

                // RangeFull
                TensorSlice::RangeFull(_) => {
                    new_shape.push_dim(dim_size);
                    new_stride.push_dim(dim_stride);
                }
            }
        }

        // Handle remained dimensions
        for dim in indices.len()..self.dims() {
            new_shape.push_dim(self.shape[dim]);
            new_stride.push_dim(self.stride[dim]);
        }

        CPUTensor {
            data: self.data.clone(), // Share
            shape: new_shape,
            stride: new_stride,
            offset: new_offset,
        }
    }

    pub fn broadcast_to(&self, target_shape: &Shape) -> Option<CPUTensor<T>> {
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

        Some(CPUTensor {
            data: self.data.clone(),
            shape: target_shape.clone(),
            stride: Shape::new(new_stride_v),
            offset: self.offset,
        })
    }
}

pub fn broadcast<T: Copy, U: Copy>(
    a: &CPUTensor<T>,
    b: &CPUTensor<U>,
) -> Option<(CPUTensor<T>, CPUTensor<U>)> {
    let target_shape: Shape = broadcast_shape(&a.shape(), &b.shape())?;
    Some((
        a.broadcast_to(&target_shape)?,
        b.broadcast_to(&target_shape)?,
    ))
}

fn tensor_fmt_recursive<T: Copy + std::fmt::Display>(
    limit: usize,
    f: &mut std::fmt::Formatter<'_>,
    tensor: &CPUTensor<T>,
    trace: &mut [usize],
    dim: usize,
) -> std::fmt::Result {
    if dim == tensor.dims() {
        return write!(f, "{}", tensor.index(trace));
    }

    write!(f, "[")?;
    if tensor.shape()[dim] <= limit * 2 {
        for i in 0..tensor.shape()[dim] {
            trace[dim] = i;
            tensor_fmt_recursive(limit, f, tensor, trace, dim + 1)?;
            write!(f, ", ")?;
        }
    } else {
        // Front
        for i in 0..limit {
            trace[dim] = i;
            tensor_fmt_recursive(limit, f, tensor, trace, dim + 1)?;
            write!(f, ", ")?;
        }

        write!(f, ", ..., ")?;

        // Back
        for i in tensor.shape()[dim] - limit..tensor.shape()[dim] {
            trace[dim] = i;
            tensor_fmt_recursive(limit, f, tensor, trace, dim + 1)?;
            write!(f, ", ")?;
        }
    }
    write!(f, "]")?;
    Ok(())
}

impl<T: Copy + std::fmt::Display> std::fmt::Display for CPUTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let limit = env::var("MYRUSTLLM_LIMIT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(3);

        let mut trace = vec![0; self.dims()];
        tensor_fmt_recursive(limit, f, self, trace.as_mut_slice(), 0)
    }
}

impl CPUTensor<f32> {
    pub fn fill_zeros(shape: &Shape) -> Self {
        CPUTensor::fill(shape, 0.0)
    }

    pub fn fill_ones(shape: &Shape) -> Self {
        CPUTensor::fill(shape, 1.0)
    }
}

impl CPUTensor<f64> {
    pub fn fill_zeros(shape: &Shape) -> Self {
        CPUTensor::fill(shape, 0.0)
    }

    pub fn fill_ones(shape: &Shape) -> Self {
        CPUTensor::fill(shape, 1.0)
    }
}

impl CPUTensor<i32> {
    pub fn fill_zeros(shape: &Shape) -> Self {
        CPUTensor::fill(shape, 0)
    }

    pub fn fill_ones(shape: &Shape) -> Self {
        CPUTensor::fill(shape, 0)
    }
}

impl CPUTensor<i64> {
    pub fn fill_zeros(shape: &Shape) -> Self {
        CPUTensor::fill(shape, 0)
    }

    pub fn fill_ones(shape: &Shape) -> Self {
        CPUTensor::fill(shape, 0)
    }
}

pub enum CPUGenericTensor {
    F32(CPUTensor<f32>),
    F64(CPUTensor<f64>),
    I32(CPUTensor<i32>),
    I64(CPUTensor<i64>),
}

impl Tensor for CPUGenericTensor {
    fn shape(&self) -> Shape {
        match self {
            CPUGenericTensor::F32(t) => t.shape(),
            CPUGenericTensor::F64(t) => t.shape(),
            CPUGenericTensor::I32(t) => t.shape(),
            CPUGenericTensor::I64(t) => t.shape(),
        }
    }
}

impl CPUGenericTensor {
    pub fn like_zeros(tensor: &CPUGenericTensor) -> Self {
        match tensor {
            CPUGenericTensor::F32(t) => {
                CPUGenericTensor::F32(CPUTensor::<f32>::fill_zeros(&t.shape()))
            }
            CPUGenericTensor::F64(t) => {
                CPUGenericTensor::F64(CPUTensor::<f64>::fill_zeros(&t.shape()))
            }
            CPUGenericTensor::I32(t) => {
                CPUGenericTensor::I32(CPUTensor::<i32>::fill_zeros(&t.shape()))
            }
            CPUGenericTensor::I64(t) => {
                CPUGenericTensor::I64(CPUTensor::<i64>::fill_zeros(&t.shape()))
            }
        }
    }
}
