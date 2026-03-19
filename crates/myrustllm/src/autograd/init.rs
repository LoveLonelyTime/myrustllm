use crate::autograd::impls::Autograd;
use crate::common::init::{
    Literal, TensorAllocInit, TensorFillInit, TensorLiteralInit, TensorOnesInit, TensorZerosInit,
};
use crate::common::ops::binary_ops::TensorAdd;
use crate::common::ops::reduce_ops::TensorAddReduce;
use crate::common::{DTypeImpl, Impl, Shape, Tensor};

impl<
    I: Impl,
    TI: DTypeImpl<I> + TensorAllocInit<I>,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> TensorAllocInit<Autograd<I, GI>> for TI
{
    fn init(shape: &Shape, device: &<Autograd<I, GI> as Impl>::Device) -> Self::Prototype {
        Self::Prototype::leaf(Tensor::new(TI::init(shape, device)))
    }
}

impl<
    I: Impl,
    TI: DTypeImpl<I> + TensorFillInit<I, T>,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
    T,
> TensorFillInit<Autograd<I, GI>, T> for TI
{
    fn init(
        value: T,
        shape: &Shape,
        device: &<Autograd<I, GI> as Impl>::Device,
    ) -> Self::Prototype {
        Self::Prototype::leaf(Tensor::new(TI::init(value, shape, device)))
    }
}

impl<
    I: Impl,
    TI: DTypeImpl<I> + TensorZerosInit<I>,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> TensorZerosInit<Autograd<I, GI>> for TI
{
    fn init(shape: &Shape, device: &<Autograd<I, GI> as Impl>::Device) -> Self::Prototype {
        Self::Prototype::leaf(Tensor::new(TI::init(shape, device)))
    }
}

impl<
    I: Impl,
    TI: DTypeImpl<I> + TensorOnesInit<I>,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> TensorOnesInit<Autograd<I, GI>> for TI
{
    fn init(shape: &Shape, device: &<Autograd<I, GI> as Impl>::Device) -> Self::Prototype {
        Self::Prototype::leaf(Tensor::new(TI::init(shape, device)))
    }
}

impl<
    I: Impl,
    TI: DTypeImpl<I> + TensorLiteralInit<I, T>,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
    T,
> TensorLiteralInit<Autograd<I, GI>, T> for TI
{
    fn init(
        literal: impl Literal<Type = T>,
        device: &<Autograd<I, GI> as Impl>::Device,
    ) -> Self::Prototype {
        Self::Prototype::leaf(Tensor::new(TI::init(literal, device)))
    }
}
