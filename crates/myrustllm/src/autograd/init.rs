use crate::{
    autograd::impls::Autograd,
    common::{Shape, Tensor, dtype::DTypeImpl, impls::Impl, init::{TensorAllocInit, TensorZerosInit}},
};

impl<I: Impl, TI: DTypeImpl<I> + TensorAllocInit<I>, GI: DTypeImpl<I>>
    TensorAllocInit<Autograd<I, GI>> for TI
{
    fn init(shape: &Shape, device: &<Autograd<I, GI> as Impl>::Device) -> Self::Prototype {
        Self::Prototype::leaf(Tensor::new(TI::init(shape, device)))
    }
}

impl<I: Impl, TI: DTypeImpl<I> + TensorZerosInit<I>, GI: DTypeImpl<I>>
    TensorZerosInit<Autograd<I, GI>> for TI
{
    fn init(shape: &Shape, device: &<Autograd<I, GI> as Impl>::Device) -> Self::Prototype {
        Self::Prototype::leaf(Tensor::new(TI::init(shape, device)))
    }
}
