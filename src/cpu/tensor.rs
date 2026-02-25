use std::cell::{Ref, RefCell, RefMut};
use std::iter::zip;
use std::rc::Rc;
use std::{env, vec};

use crate::cpu::interface;
use crate::cpu::literal::Literal;
use crate::cpu::mem::CPUMemory;
use crate::cpu::shape::{Shape, broadcast_shape, create_contiguous_stride};
use crate::cpu::slice::TensorIndex;

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
    /// Create a new CPU tensor.
    pub fn new(data: CPUMemory<T>, shape: &Shape, stride: &Shape, offset: usize) -> Self {
        CPUTensor {
            data: Rc::new(RefCell::new(data)),
            shape: shape.clone(),
            stride: stride.clone(),
            offset,
        }
    }

    /// Create a new CPU tensor with shape `shape`.
    ///
    /// The new tensor will be filled with dirty data.
    pub fn from_shape(shape: &Shape) -> Self {
        CPUTensor::new(
            CPUMemory::new(shape.numel()),
            shape,
            &create_contiguous_stride(&shape),
            0,
        )
    }

    /// Create a new CPU tensor filled with `input`.
    pub fn fill(shape: &Shape, input: T) -> Self {
        let mut mem = CPUMemory::<T>::new(shape.numel());

        // TODO: Do we need OpenMP?
        unsafe {
            for i in 0..mem.size() {
                mem.as_mut_ptr().add(i).write(input.into());
            }
        };

        CPUTensor::new(mem, shape, &create_contiguous_stride(&shape), 0)
    }

    /// Create a new CPU tensor from the array `input`
    ///
    /// # Examples
    ///
    /// ```
    /// use myrustllm::cpu::tensor::CPUTensor;
    ///
    /// let tensor = CPUTensor::<f32>::from_literal([
    ///     [1.0, 1.0],
    ///     [2.0, 2.0]
    /// ]);
    /// ```
    pub fn from_literal<U: Literal<T>>(input: U) -> Self {
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

    /// Create a new CPU scalar.
    pub fn scalar(input: T) -> Self {
        let shape = Shape::scalar();
        let stride = create_contiguous_stride(&shape);
        let mut mem = CPUMemory::<T>::new(1);
        unsafe {
            *mem.as_mut_ptr() = input;
        };

        CPUTensor::new(mem, &shape, &stride, 0)
    }

    /// Borrow a ref of data.
    pub fn borrow(&self) -> Ref<'_, CPUMemory<T>> {
        self.data.borrow()
    }

    /// Borrow a mutable ref of data.
    pub fn borrow_mut(&self) -> RefMut<'_, CPUMemory<T>> {
        self.data.borrow_mut()
    }

    /// Return the stride of the tensor.
    pub fn stride(&self) -> Shape {
        self.stride.clone()
    }

    /// Return the offset of the tensor.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Return the total stride of the tensor.
    pub fn get_memory_stride(&self) -> usize {
        zip(self.shape().iter(), self.stride().iter())
            .fold(1, |acc, (dim, stride)| acc + (dim - 1) * stride)
    }

    /// Return the scalar.
    ///
    /// If the tensor is a scalar, it will the scalar.
    pub fn into_scalar(&self) -> T {
        assert!(self.is_scalar(), "The tensor is not a scalar");
        unsafe { *self.borrow_mut().as_ptr().add(self.offset()) }
    }

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
        let mut new_shape_v = Vec::new();
        let mut new_stride_v = Vec::new();
        let mut new_offset = self.offset;

        fn _neg_index_to_pos_index(index: isize, dim: usize) -> usize {
            if index >= 0 {
                index as usize
            } else {
                dim - ((-index) as usize)
            }
        }

        // Step 1: Check `Full`
        let (indices, mut indices_i, mut dim_i, incr) = if !indices.is_empty() {
            if indices[0] == TensorIndex::Full {
                (&indices[1..], indices.len() - 1, self.dims(), false)
            } else if indices[indices.len() - 1] == TensorIndex::Full {
                (&indices[..indices.len() - 1], 1, 1, true)
            } else {
                (indices, 1, 1, true)
            }
        } else {
            (indices, 1, 1, true)
        };

        // Step 2: Match
        while indices_i > 0 && indices_i <= indices.len() && dim_i > 0 && dim_i <= self.dims() {
            let dim_size = self.shape[dim_i - 1];
            let dim_stride = self.stride[dim_i - 1];

            match indices[indices_i - 1] {
                // Index
                TensorIndex::Index(_i) => {
                    let i = _neg_index_to_pos_index(_i, dim_size);
                    assert!(
                        i < dim_size,
                        "Index {} out of bounds of dimension {} with size {}.",
                        _i,
                        dim_i - 1,
                        dim_size
                    );
                    new_offset += i * dim_stride;
                    if incr {
                        dim_i = dim_i + 1;
                    } else {
                        dim_i = dim_i - 1;
                    };
                }

                // Range
                TensorIndex::Range(_start, _end) => {
                    let (start, end) = (
                        _neg_index_to_pos_index(_start, dim_size),
                        _neg_index_to_pos_index(_end, dim_size),
                    );
                    assert!(
                        start < dim_size && end <= dim_size && start < end,
                        "Range {:?} out of bounds of dimension {} with size {}.",
                        TensorIndex::Range(_start, _end),
                        dim_i - 1,
                        dim_size
                    );
                    new_offset += start * dim_stride;
                    new_shape_v.push(end - start);
                    new_stride_v.push(dim_stride);
                    if incr {
                        dim_i = dim_i + 1;
                    } else {
                        dim_i = dim_i - 1;
                    };
                }

                // RangeFrom
                TensorIndex::RangeFrom(_start) => {
                    let start = _neg_index_to_pos_index(_start, dim_size);
                    assert!(
                        start < dim_size,
                        "Range {:?} out of bounds of dimension {} with size {}.",
                        TensorIndex::RangeFrom(_start),
                        dim_i - 1,
                        dim_size
                    );
                    new_offset += start * dim_stride;
                    new_shape_v.push(dim_size - start);
                    new_stride_v.push(dim_stride);
                    if incr {
                        dim_i = dim_i + 1;
                    } else {
                        dim_i = dim_i - 1;
                    };
                }

                // RangeTo
                TensorIndex::RangeTo(_end) => {
                    let end = _neg_index_to_pos_index(_end, dim_size);
                    assert!(
                        end > 0 && end <= dim_size,
                        "Range {:?} out of bounds of dimension {} with size {}.",
                        TensorIndex::RangeTo(_end),
                        dim_i - 1,
                        dim_size
                    );
                    new_shape_v.push(end);
                    new_stride_v.push(dim_stride);
                    if incr {
                        dim_i = dim_i + 1;
                    } else {
                        dim_i = dim_i - 1;
                    };
                }

                // RangeFull
                TensorIndex::RangeFull => {
                    new_shape_v.push(dim_size);
                    new_stride_v.push(dim_stride);
                    if incr {
                        dim_i = dim_i + 1;
                    } else {
                        dim_i = dim_i - 1;
                    };
                }

                // RangeInclusive
                TensorIndex::RangeInclusive(_start, _end) => {
                    let (start, end) = (
                        _neg_index_to_pos_index(_start, dim_size),
                        _neg_index_to_pos_index(_end, dim_size),
                    );
                    assert!(
                        start < dim_size && end < dim_size && start <= end,
                        "Range {:?} out of bounds of dimension {} with size {}.",
                        TensorIndex::RangeInclusive(_start, _end),
                        dim_i - 1,
                        dim_size
                    );
                    new_offset += start * dim_stride;
                    new_shape_v.push(end - start + 1);
                    new_stride_v.push(dim_stride);
                    if incr {
                        dim_i = dim_i + 1;
                    } else {
                        dim_i = dim_i - 1;
                    };
                }

                // RangeToInclusive
                TensorIndex::RangeToInclusive(_end) => {
                    let end = _neg_index_to_pos_index(_end, dim_size);
                    assert!(
                        end < dim_size,
                        "Range {:?} out of bounds of dimension {} with size {}.",
                        TensorIndex::RangeToInclusive(_end),
                        dim_i - 1,
                        dim_size
                    );
                    new_shape_v.push(end + 1);
                    new_stride_v.push(dim_stride);
                    if incr {
                        dim_i = dim_i + 1;
                    } else {
                        dim_i = dim_i - 1;
                    };
                }

                // Expand
                TensorIndex::Expand => {
                    new_shape_v.push(1);
                    new_stride_v.push(0);
                }

                TensorIndex::Full => {
                    panic!("Inner `Full` is not supported.");
                }
            }
            if incr {
                indices_i = indices_i + 1;
            } else {
                indices_i = indices_i - 1;
            };
        }

        assert!(
            (incr && indices_i == indices.len() + 1) || (!incr && indices_i == 0),
            "Indices aren't exhausted."
        );

        // Step 3: Handle remained dimensions
        while dim_i > 0 && dim_i <= self.dims() {
            new_shape_v.push(self.shape[dim_i - 1]);
            new_stride_v.push(self.stride[dim_i - 1]);
            if incr {
                dim_i = dim_i + 1;
            } else {
                dim_i = dim_i - 1;
            };
        }

        if !incr {
            new_shape_v.reverse();
            new_stride_v.reverse();
        }

        CPUTensor {
            data: self.data.clone(), // Share
            shape: Shape::new(new_shape_v),
            stride: Shape::new(new_stride_v),
            offset: new_offset,
        }
    }

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
            data: self.data.clone(), // Share
            shape: target_shape.clone(),
            stride: Shape::new(new_stride_v),
            offset: self.offset,
        })
    }

    /// Return a new tensor with the same data as the tensor `&self` but of a different shape `new_shape`.
    ///
    /// For a tensor to be viewed, the new view size must be compatible with its original size and stride.
    /// In other words, the new view size must completely merge and split the contiguous subspaces derived from the tensor `&self`.
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
    pub fn view(&self, new_shape: &Shape) -> Self {
        let mut new_shape = new_shape.clone();

        // Check inferred dim
        let mut inferred_dim = None;
        let mut known_size = 1;
        for (i, &dim) in new_shape.iter().enumerate() {
            if dim == 0 {
                if inferred_dim.is_some() {
                    panic!("The number of auto-inferred dims is greater than 1.");
                }

                inferred_dim = Some(i);
            } else {
                known_size *= dim;
            }
        }

        if let Some(i) = inferred_dim {
            if self.shape().numel() % known_size != 0 {
                panic!("Cannot infer the auto-inferred dim.");
            }

            new_shape[i] = self.shape().numel() / known_size;
            known_size *= new_shape[i];
        }

        if self.shape().numel() != known_size {
            panic!(
                "The new number {} of elements is not equal to the original number {}.",
                known_size,
                self.shape().numel()
            );
        }

        // Merge
        let mut merged_shape_v = Vec::new();
        let mut merged_stride_v = Vec::new();

        let mut i = self.shape().len();

        if i == 0 {
            // Scalar
            merged_shape_v.push(1);
            merged_stride_v.push(1);
        }

        while i > 0 {
            // Tensor
            let mut block_dim = self.shape()[i - 1];
            let mut block_stride = self.stride()[i - 1];
            i -= 1;

            while i > 0 && self.stride()[i - 1] == self.stride()[i] * self.shape()[i] {
                block_dim *= self.shape()[i - 1];
                block_stride = self.stride()[i];
                i -= 1;
            }
            merged_shape_v.push(block_dim);
            merged_stride_v.push(block_stride);
        }

        merged_shape_v.reverse();
        merged_stride_v.reverse();

        // Split
        let mut new_stride = Shape::new(vec![0; new_shape.len()]);
        let mut block_i = merged_shape_v.len();
        let mut new_i = new_shape.len();

        while block_i > 0 {
            let block_dim = merged_shape_v[block_i - 1];
            let block_stride = merged_stride_v[block_i - 1];
            block_i -= 1;

            let mut acc = 1;
            let mut dims = Vec::new();
            while new_i > 0 && acc <= block_dim {
                acc *= new_shape[new_i - 1];
                dims.push(new_i - 1);
                new_i -= 1;
            }

            if acc != block_dim {
                panic!("Cannot split old subspaces.");
            }

            let mut running_stride = block_stride;
            for dim in dims {
                new_stride[dim] = running_stride;
                running_stride *= new_shape[dim];
            }
        }

        if new_i != 0 {
            panic!("Cannot split old subspaces.");
        }

        CPUTensor {
            data: self.data.clone(), // Share
            shape: new_shape,
            stride: new_stride,
            offset: self.offset(),
        }
    }

    /// Return a tensor with the same data copyed from its original data, but with the specified shape `new_shape`.
    ///
    /// The number of elements must be equal.
    pub fn reshape(&self, new_shape: &Shape) -> Self {
        let mut new_shape = new_shape.clone();

        // Check inferred dim
        let mut inferred_dim = None;
        let mut known_size = 1;
        for (i, &dim) in new_shape.iter().enumerate() {
            if dim == 0 {
                if inferred_dim.is_some() {
                    panic!("The number of auto-inferred dims is greater than 1.");
                }

                inferred_dim = Some(i);
            } else {
                known_size *= dim;
            }
        }

        if let Some(i) = inferred_dim {
            if self.shape().numel() % known_size != 0 {
                panic!("Cannot infer the auto-inferred dim.");
            }

            new_shape[i] = self.shape().numel() / known_size;
            known_size *= new_shape[i];
        }

        if self.shape().numel() != known_size {
            panic!(
                "The new number {} of elements is not equal to the original number {}.",
                known_size,
                self.shape().numel()
            );
        }

        // Cpoy
        let out = CPUTensor::from_shape(&new_shape);
        unsafe {
            let self_ptr = self.borrow().as_ptr() as *const libc::c_float;
            let out_ptr = out.borrow_mut().as_mut_ptr() as *mut libc::c_float;
            interface::cpu_tensor_copy_f32(
                self_ptr.add(self.offset()),
                out_ptr.add(out.offset()),
                self.stride().as_ptr(),
                out.stride().as_ptr(),
                self.shape().as_ptr(),
                out.shape().as_ptr(),
                self.dims(),
                out.dims(),
                self.numel(),
            );
        }

        out
    }

    /// Permute the dimensions of the tensor.
    ///
    /// For `dims`, each number from `0..dims.len()` must appear just once.
    pub fn permute(&self, dims: &[usize]) -> Self {
        // Check
        assert!(
            dims.len() == self.dims(),
            "The size of dims must be {}, but got {}.",
            self.dims(),
            dims.len()
        );

        let mut checklist = vec![false; self.dims()];
        for &i in dims {
            assert!(
                i < self.dims(),
                "Index {} is out of bounds of the dimensions with size {}.",
                i,
                self.dims()
            );
            checklist[i] = true;
        }
        assert!(checklist.iter().all(|&x| x), "Invalid dims: {:?}.", dims);

        // Permute
        let mut new_shape = Shape::new(vec![0; self.dims()]);
        let mut new_stride = Shape::new(vec![0; self.dims()]);
        for (new_i, &i) in dims.iter().enumerate() {
            new_shape[new_i] = self.shape()[i];
            new_stride[new_i] = self.stride()[i];
        }

        CPUTensor {
            data: self.data.clone(), // Share
            shape: new_shape,
            stride: new_stride,
            offset: self.offset,
        }
    }
}

// `Clone` for `CPUTensor<T>` doesn't mean copying data, but creates a view.
impl<T: Copy> Clone for CPUTensor<T> {
    fn clone(&self) -> Self {
        CPUTensor {
            data: self.data.clone(), // Share
            shape: self.shape(),
            stride: self.stride(),
            offset: self.offset,
        }
    }
}

// ---------- Utils ----------

/// Try broadcast two cpu tensors mutually.
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

// This can let users use `println!` for CPUTensor
impl<T: Copy + std::fmt::Display> std::fmt::Display for CPUTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let limit = env::var("MYRUSTLLM_DISPLAY_LIMIT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(3);

        let mut trace = vec![0; self.dims()];

        fn _fmt_recursive<T: Copy + std::fmt::Display>(
            limit: usize,
            f: &mut std::fmt::Formatter<'_>,
            tensor: &CPUTensor<T>,
            trace: &mut [isize],
            dim: usize,
        ) -> std::fmt::Result {
            if dim == tensor.dims() {
                let indices: Vec<TensorIndex> =
                    trace.iter().map(|&i| TensorIndex::Index(i)).collect();
                return write!(f, "{}", tensor.slice(&indices).into_scalar());
            }

            write!(f, "[")?;
            if tensor.shape()[dim] <= limit * 2 {
                for i in 0..tensor.shape()[dim] {
                    trace[dim] = i as isize;
                    _fmt_recursive(limit, f, tensor, trace, dim + 1)?;
                    write!(f, ", ")?;
                }
            } else {
                // Front
                for i in 0..limit {
                    trace[dim] = i as isize;
                    _fmt_recursive(limit, f, tensor, trace, dim + 1)?;
                    write!(f, ", ")?;
                }

                write!(f, ", ..., ")?;

                // Back
                for i in tensor.shape()[dim] - limit..tensor.shape()[dim] {
                    trace[dim] = i as isize;
                    _fmt_recursive(limit, f, tensor, trace, dim + 1)?;
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")?;
            Ok(())
        }

        _fmt_recursive(limit, f, self, trace.as_mut_slice(), 0)
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

/// CPUGenericTensor
/// Compared to `CPUTensor<T>`, the data type of `CPUGenericTensor` is dispatched dynamically.
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
