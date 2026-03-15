use crate::{
    autograd::impls::Autograd,
    common::{dtype::DTypeImpl, impls::Impl, ops::binary_ops::TensorAddImpl},
};

impl<I: Impl, GI: DTypeImpl<I>, TI: DTypeImpl<I> + TensorAddImpl<I, Rhs>, Rhs: DTypeImpl<I>>
    TensorAddImpl<Autograd<I, GI>, Rhs> for TI
{
    type Output = TI::Output;
    fn add(
        lhs: &Self::Prototype,
        rhs: &<Rhs as DTypeImpl<Autograd<I, GI>>>::Prototype,
    ) -> <Self::Output as DTypeImpl<Autograd<I, GI>>>::Prototype {
        todo!()
    }
}
