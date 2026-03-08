use crate::cpu::mem::RawData;
use crate::common::Shape;

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
        let mut shape_v = vec![N];
        if N > 0 {
            shape_v.extend(self[0].shape().iter());
        }
        shape_v.into()
    }

    fn flatten(&self, out: &mut Vec<T>) {
        for item in self {
            item.flatten(out);
        }
    }
}

// The implementation for raw data
impl<T: RawData> Literal<T> for T {
    fn shape(&self) -> Shape {
        Shape::scalar()
    }

    fn flatten(&self, out: &mut Vec<T>) {
        out.push(*self);
    }
}
