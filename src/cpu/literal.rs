use crate::cpu::shape::Shape;

/// Literal
pub trait Literal<T> {
    /// Return the original shape
    fn shape(&self) -> Shape;

    /// Flatten the literal and append to `out`
    fn flatten(&self, out: &mut Vec<T>);
}

// The recursive implementation for array
impl<T, U: Literal<T>, const N: usize> Literal<T> for [U; N] {
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

// The implementations for scalar
macro_rules! literal_scalar {
    ($t:ty) => {
        impl Literal<$t> for $t {
            fn shape(&self) -> Shape {
                Shape::scalar()
            }

            fn flatten(&self, out: &mut Vec<$t>) {
                out.push(*self);
            }
        }
    };
}

literal_scalar!(f32);
literal_scalar!(f64);
literal_scalar!(i32);
literal_scalar!(i64);
