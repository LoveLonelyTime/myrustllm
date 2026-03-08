use std::ops::Deref;
use std::rc::Rc;

/// Metadata for shape, stride.
#[derive(Clone, PartialEq, Eq)]
pub struct Shape(Rc<[usize]>);

impl Shape {
    /// Create a new shape for scalar.
    pub fn scalar() -> Self {
        Shape(Rc::new([]))
    }

    /// Return the number of elements along the shape.
    pub fn numel(&self) -> usize {
        if self.is_scalar() {
            1
        } else {
            self.iter().product()
        }
    }

    /// Is the shape for scalar?
    pub fn is_scalar(&self) -> bool {
        self.len() == 0
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
        let mut stride_v = vec![1; shape.len()];
        if !shape.is_scalar() {
            for i in (0..shape.len() - 1).rev() {
                stride_v[i] = stride_v[i + 1] * shape[i + 1];
            }
        }
        stride_v.into()
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
        return Some(result_shape_v.into());
    }
}

impl Deref for Shape {
    type Target = Rc<[usize]>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Into<Rc<[usize]>>> From<T> for Shape {
    fn from(value: T) -> Self {
        Shape(value.into())
    }
}

impl std::fmt::Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_scalar() {
            write!(f, "Shape(scalar)")
        } else {
            write!(f, "Shape({:?})", self.0)
        }
    }
}
