use std::usize;

/// Metadata for shape, stride, ...
#[derive(Debug, Clone)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// Create a new shape.
    pub fn new(dims: Vec<usize>) -> Self {
        Shape(dims)
    }

    /// Create a new shape for scalar.
    pub fn scalar() -> Self {
        Shape(Vec::new())
    }

    /// Return the number of elements along the shape.
    pub fn numel(&self) -> usize {
        if self.is_scalar() {
            1
        } else {
            self.0.iter().product()
        }
    }

    /// Push a new dim into the shape.
    pub fn push_dim(&mut self, dim: usize) {
        self.0.push(dim);
    }

    /// Extend the shape.
    pub fn extend_dim(&mut self, dims: &Shape) {
        self.0.extend(&dims.0);
    }

    /// Return the dimension of the shape.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Is the shape for scalar?
    pub fn is_scalar(&self) -> bool {
        self.len() == 0
    }

    /// Return an iter.
    pub fn iter(&self) -> std::slice::Iter<'_, usize> {
        self.0.iter()
    }

    /// Return a const pointer.
    pub fn as_ptr(&self) -> *const usize {
        self.0.as_ptr()
    }

    /// Return a mutable pointer.
    pub fn as_mut_ptr(&mut self) -> *mut usize {
        self.0.as_mut_ptr()
    }
}

impl PartialEq for Shape {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl std::ops::Index<usize> for Shape {
    type Output = usize;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index]
    }
}

impl std::ops::IndexMut<usize> for Shape {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.0[index]
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_scalar() {
            write!(f, "Shape(scalar)")?;
        } else {
            write!(f, "Shape(")?;
            for (i, num) in self.0.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", num)?;
            }
            write!(f, ")")?;
        }
        Ok(())
    }
}

/// Create a contiguous stride along the shape.
///
/// # Example
///
/// ```
/// use myrustllm::cpu::shape::{Shape, create_contiguous_stride};
///
/// let shape = Shape::new(vec![2, 3, 4]);
/// let contiguous_stride = create_contiguous_stride(&shape);
/// assert_eq!(contiguous_stride, Shape::new(vec![12, 4, 1]));
/// ```
///
/// For the above example, 1 = 1, 4 = 1 * 4, 12 = 3 * 4.
pub fn create_contiguous_stride(shape: &Shape) -> Shape {
    let mut stride = Shape::new(vec![1; shape.len()]);

    if shape.is_scalar() {
        stride
    } else {
        for i in (0..shape.len() - 1).rev() {
            stride[i] = stride[i + 1] * shape[i + 1];
        }

        stride
    }
}

pub fn broadcast_shape(shape_a: &Shape, shape_b: &Shape) -> Option<Shape> {
    let max_dims = std::cmp::max(shape_a.len(), shape_b.len());

    let mut result_shape_v = Vec::with_capacity(max_dims);

    for i in 0..max_dims {
        let d_a = if i < shape_a.len() {
            shape_a[shape_a.len() - 1 - i]
        } else {
            1
        };
        let d_b = if i < shape_b.len() {
            shape_b[shape_b.len() - 1 - i]
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
