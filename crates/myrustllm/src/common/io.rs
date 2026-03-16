use crate::common::dtype::{F32, F64, I32, I64};
use crate::common::{DType, DTypeOf, Shape};
use std::io::Read;

/// `TensorRawData` hold a reference of raw data (byte source), which can be used to init a tensor.
///
/// The byte source must follow the rules declared by `DType`, e.g. endian, size, and alignment.
pub struct TensorRawData {
    // TODO: Seek?
    pub source: Box<dyn Read>,
    pub shape: Shape,
    pub dtype: DType,
}

/// Rust scalars and rust arrays with scalars can be considered as a literal.
pub trait Literal {
    const DTYPE: DType;
    fn shape(&self) -> Shape;
    fn data(&self) -> Vec<u8>;
}

impl<T: Literal> From<T> for TensorRawData {
    fn from(value: T) -> Self {
        let shape = value.shape();
        let data = value.data();
        TensorRawData {
            source: Box::new(std::io::Cursor::new(data)),
            shape,
            dtype: T::DTYPE,
        }
    }
}

macro_rules! register_scalar_literal {
    ($dt: ty, $rt: ty) => {
        impl Literal for $rt {
            const DTYPE: DType = <$dt as DTypeOf>::DTYPE;

            fn shape(&self) -> Shape {
                Shape::scalar()
            }

            fn data(&self) -> Vec<u8> {
                // Little-endian
                Vec::from(self.to_le_bytes())
            }
        }
    };
}

register_scalar_literal!(F32, f32);
register_scalar_literal!(F64, f64);
register_scalar_literal!(I32, i32);
register_scalar_literal!(I64, i64);

impl<T: Literal, const N: usize> Literal for [T; N] {
    const DTYPE: DType = T::DTYPE;

    fn shape(&self) -> Shape {
        assert!(N > 0, "Empty literal isn't supported.");
        let mut shape_v = vec![N];
        shape_v.extend(self[0].shape().iter());
        Shape::from(shape_v)
    }

    fn data(&self) -> Vec<u8> {
        let mut vec = Vec::new();
        for item in self {
            vec.extend(item.data());
        }
        vec
    }
}
