//! This mod (init) defines how to init a tensor.
//!
//! List:
//! - TensorAllocInit: alloc
//! - TensorFillInit: fill
//! - TensorZerosInit: zeros
//! - TensorOnesInit: ones
//! - TensorLiterialInit: from_literal

use crate::common::{DTypeImpl, Impl, Shape, Tensor};

/// Tensor alloc init implementation.
pub trait TensorAllocInit<I: Impl>: DTypeImpl<I> {
    fn init(shape: &Shape, device: &I::Device) -> Self::Prototype;
}

impl<I: Impl, TI: DTypeImpl<I> + TensorAllocInit<I>> Tensor<I, TI> {
    /// Allocate a tensor without any initial data.
    ///
    /// Alloc just allocates a memory for a tensor, which may contain dirty data.
    pub fn alloc(shape: &Shape, device: &I::Device) -> Self {
        Tensor::new(TI::init(shape, device))
    }
}

/// Tensor fill init implementation.
pub trait TensorFillInit<I: Impl, T>: DTypeImpl<I> {
    fn init(value: T, shape: &Shape, device: &I::Device) -> Self::Prototype;
}

impl<I: Impl, TI: DTypeImpl<I>> Tensor<I, TI> {
    /// Create and fill a tensor with the given value.
    pub fn fill<T>(value: T, shape: &Shape, device: &I::Device) -> Self
    where
        TI: TensorFillInit<I, T>,
    {
        Tensor::new(TI::init(value, shape, device))
    }
}

/// Tensor zeros init implementation.
pub trait TensorZerosInit<I: Impl>: DTypeImpl<I> {
    fn init(shape: &Shape, device: &I::Device) -> Self::Prototype;
}

impl<I: Impl, TI: DTypeImpl<I> + TensorZerosInit<I>> Tensor<I, TI> {
    /// Create a tensor with zero data.
    pub fn zeros(shape: &Shape, device: &I::Device) -> Self {
        Tensor::new(TI::init(shape, device))
    }
}

/// Tensor ones init implementation.
pub trait TensorOnesInit<I: Impl>: DTypeImpl<I> {
    fn init(shape: &Shape, device: &I::Device) -> Self::Prototype;
}

impl<I: Impl, TI: DTypeImpl<I> + TensorOnesInit<I>> Tensor<I, TI> {
    /// Create a tensor with one data.
    pub fn ones(shape: &Shape, device: &I::Device) -> Self {
        Tensor::new(TI::init(shape, device))
    }
}

/// Rust scalars and rust arrays with scalars can be considered as a literal.
pub trait Literal {
    type Type;
    fn shape(&self) -> Shape;
    fn data(&self) -> Vec<Self::Type>;
}

macro_rules! impl_literal_scalar {
    ($($t:ty),*) => {
        $(
            impl Literal for $t {
                type Type = $t;

                fn shape(&self) -> Shape {
                    Shape::scalar()
                }

                fn data(&self) -> Vec<Self::Type> {
                    vec![*self]
                }
            }
        )*
    };
}

impl_literal_scalar!(f32, f64, i32, i64);

impl<T: Literal, const N: usize> Literal for [T; N] {
    type Type = T::Type;

    fn shape(&self) -> Shape {
        assert!(N > 0, "Empty literal isn't supported.");
        let mut shape_v = vec![N];
        shape_v.extend(self[0].shape().iter());
        Shape::from(shape_v)
    }

    fn data(&self) -> Vec<Self::Type> {
        assert!(N > 0, "Empty literal isn't supported.");
        let mut vec = Vec::new();
        for item in self {
            vec.extend(item.data());
        }
        vec
    }
}

// Tensor literal init implementation.
pub trait TensorLiteralInit<I: Impl, U>: DTypeImpl<I> {
    fn init(literal: impl Literal<Type = U>, device: &I::Device) -> Self::Prototype;
}

impl<I: Impl, TI: DTypeImpl<I>> Tensor<I, TI> {
    /// Create and fill a tensor with the given value.
    pub fn from_literal<T>(literal: impl Literal<Type = T>, device: &I::Device) -> Self
    where
        TI: TensorLiteralInit<I, T>,
    {
        Tensor::new(TI::init(literal, device))
    }
}
