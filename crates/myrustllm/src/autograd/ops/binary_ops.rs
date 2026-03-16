use crate::{
    autograd::impls::Autograd,
    common::{dtype::DTypeImpl, impls::Impl, ops::binary_ops::TensorAdd},
};

impl<I: Impl, GI: DTypeImpl<I>, TI: DTypeImpl<I> + TensorAdd<I, Rhs>, Rhs: DTypeImpl<I>>
    TensorAdd<Autograd<I, GI>, Rhs> for TI
{
    type Output = TI::Output;
    fn add(
        lhs: &Self::Prototype,
        rhs: &<Rhs as DTypeImpl<Autograd<I, GI>>>::Prototype,
    ) -> <Self::Output as DTypeImpl<Autograd<I, GI>>>::Prototype {
        todo!()
    }
}
