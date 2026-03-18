use crate::common::init::{
    Literal, TensorAllocInit, TensorFillInit, TensorLiteralInit, TensorOnesInit, TensorZerosInit,
};
use crate::common::ops::cast::TensorCopy;
use crate::common::shape::create_contiguous_stride;
use crate::common::{DTypeImpl, Impl, Shape};
use crate::cpu::impls::{CPU, CPUTensorPrototype};
use crate::cpu::mem::CPUMemory;
use num_traits::{One, Zero};

impl<T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<U>>, U> TensorAllocInit<CPU> for T {
    fn init(shape: &Shape, _device: &<CPU as Impl>::Device) -> Self::Prototype {
        CPUTensorPrototype::alloc(shape)
    }
}

impl<T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<U>> + TensorCopy<CPU, T>, U>
    TensorFillInit<CPU, U> for T
{
    fn init(value: U, shape: &Shape, device: &<CPU as Impl>::Device) -> Self::Prototype {
        let mut out = <T as TensorAllocInit<CPU>>::init(shape, device);
        <T as TensorCopy<CPU, T>>::copy(
            &mut out,
            &CPUTensorPrototype::new(
                CPUMemory::from([value]).into(),
                &Shape::scalar(),
                &create_contiguous_stride(&Shape::scalar()),
                0,
            ),
        );

        out
    }
}

impl<T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<U>> + TensorFillInit<CPU, U>, U: Zero>
    TensorZerosInit<CPU> for T
{
    fn init(shape: &Shape, device: &<CPU as Impl>::Device) -> Self::Prototype {
        T::init(U::zero(), shape, device)
    }
}

impl<T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<U>> + TensorFillInit<CPU, U>, U: One>
    TensorOnesInit<CPU> for T
{
    fn init(shape: &Shape, device: &<CPU as Impl>::Device) -> Self::Prototype {
        T::init(U::one(), shape, device)
    }
}

impl<T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<U>>, U> TensorLiteralInit<CPU, U> for T {
    fn init(literal: impl Literal<Type = U>, _device: &<CPU as Impl>::Device) -> Self::Prototype {
        let mem = CPUMemory::from(literal.data());
        CPUTensorPrototype::new(
            mem.into(),
            &literal.shape(),
            &create_contiguous_stride(&literal.shape()),
            0,
        )
    }
}
