use crate::common::init::TensorAllocInit;
use crate::common::ops::binary_ops::{TensorDiv, TensorMul, TensorSub};
use crate::common::ops::{
    binary_ops::TensorAdd,
    cast::TensorCast,
    promote::Promote,
    view::{TensorBroadcast, broadcast_prot},
};
use crate::common::{DTypeImpl, TensorPrototype};
use crate::cpu::impls::{CPU, CPUTensorPrototype};
use crate::cpu::interface;
use crate::cpu::interface::IntoInterface;

impl<Lhs, Rhs, L, R, O> TensorAdd<CPU, Rhs> for Lhs
where
    Lhs: Promote<Rhs> + DTypeImpl<CPU, Prototype = CPUTensorPrototype<L>>,
    Rhs: DTypeImpl<CPU, Prototype = CPUTensorPrototype<R>>,
    Lhs: TensorCast<CPU, <Lhs as Promote<Rhs>>::Output>,
    Rhs: TensorCast<CPU, <Lhs as Promote<Rhs>>::Output>,
    <Lhs as Promote<Rhs>>::Output: DTypeImpl<CPU, Prototype = CPUTensorPrototype<O>>
        + TensorBroadcast<CPU>
        + TensorAllocInit<CPU>,
    CPUTensorPrototype<L>: TensorPrototype<CPU>,
    CPUTensorPrototype<R>: TensorPrototype<CPU>,
    CPUTensorPrototype<O>: TensorPrototype<CPU>,
{
    type Output = <Lhs as Promote<Rhs>>::Output;

    fn add(
        lhs: &Self::Prototype,
        rhs: &Rhs::Prototype,
    ) -> <Self::Output as DTypeImpl<CPU>>::Prototype {

        let (lhs, rhs) = broadcast_prot::<CPU, Self::Output, Self::Output>(
            &<Lhs as TensorCast<CPU, Self::Output>>::cast(lhs),
            &<Rhs as TensorCast<CPU, Self::Output>>::cast(rhs),
        )
        .unwrap_or_else(|| {
            panic!(
                "Two tensors with shapes ({:?}, {:?}) cannot be broadcast.",
                lhs.shape(),
                rhs.shape()
            )
        });

        let out = <Self::Output as TensorAllocInit<CPU>>::init(&lhs.shape(), &Default::default());

        unsafe {
            interface::cpu_tensor_add(
                out.into_interface(),
                lhs.into_interface(),
                rhs.into_interface(),
            );
        };

        out
    }
}


impl<Lhs, Rhs, L, R, O> TensorSub<CPU, Rhs> for Lhs
where
    Lhs: Promote<Rhs> + DTypeImpl<CPU, Prototype = CPUTensorPrototype<L>>,
    Rhs: DTypeImpl<CPU, Prototype = CPUTensorPrototype<R>>,
    Lhs: TensorCast<CPU, <Lhs as Promote<Rhs>>::Output>,
    Rhs: TensorCast<CPU, <Lhs as Promote<Rhs>>::Output>,
    <Lhs as Promote<Rhs>>::Output: DTypeImpl<CPU, Prototype = CPUTensorPrototype<O>>
        + TensorBroadcast<CPU>
        + TensorAllocInit<CPU>,
    CPUTensorPrototype<L>: TensorPrototype<CPU>,
    CPUTensorPrototype<R>: TensorPrototype<CPU>,
    CPUTensorPrototype<O>: TensorPrototype<CPU>,
{
    type Output = <Lhs as Promote<Rhs>>::Output;

    fn sub(
        lhs: &Self::Prototype,
        rhs: &Rhs::Prototype,
    ) -> <Self::Output as DTypeImpl<CPU>>::Prototype {

        let (lhs, rhs) = broadcast_prot::<CPU, Self::Output, Self::Output>(
            &<Lhs as TensorCast<CPU, Self::Output>>::cast(lhs),
            &<Rhs as TensorCast<CPU, Self::Output>>::cast(rhs),
        )
        .unwrap_or_else(|| {
            panic!(
                "Two tensors with shapes ({:?}, {:?}) cannot be broadcast.",
                lhs.shape(),
                rhs.shape()
            )
        });

        let out = <Self::Output as TensorAllocInit<CPU>>::init(&lhs.shape(), &Default::default());

        unsafe {
            interface::cpu_tensor_sub(
                out.into_interface(),
                lhs.into_interface(),
                rhs.into_interface(),
            );
        };

        out
    }
}


impl<Lhs, Rhs, L, R, O> TensorMul<CPU, Rhs> for Lhs
where
    Lhs: Promote<Rhs> + DTypeImpl<CPU, Prototype = CPUTensorPrototype<L>>,
    Rhs: DTypeImpl<CPU, Prototype = CPUTensorPrototype<R>>,
    Lhs: TensorCast<CPU, <Lhs as Promote<Rhs>>::Output>,
    Rhs: TensorCast<CPU, <Lhs as Promote<Rhs>>::Output>,
    <Lhs as Promote<Rhs>>::Output: DTypeImpl<CPU, Prototype = CPUTensorPrototype<O>>
        + TensorBroadcast<CPU>
        + TensorAllocInit<CPU>,
    CPUTensorPrototype<L>: TensorPrototype<CPU>,
    CPUTensorPrototype<R>: TensorPrototype<CPU>,
    CPUTensorPrototype<O>: TensorPrototype<CPU>,
{
    type Output = <Lhs as Promote<Rhs>>::Output;

    fn mul(
        lhs: &Self::Prototype,
        rhs: &Rhs::Prototype,
    ) -> <Self::Output as DTypeImpl<CPU>>::Prototype {

        let (lhs, rhs) = broadcast_prot::<CPU, Self::Output, Self::Output>(
            &<Lhs as TensorCast<CPU, Self::Output>>::cast(lhs),
            &<Rhs as TensorCast<CPU, Self::Output>>::cast(rhs),
        )
        .unwrap_or_else(|| {
            panic!(
                "Two tensors with shapes ({:?}, {:?}) cannot be broadcast.",
                lhs.shape(),
                rhs.shape()
            )
        });

        let out = <Self::Output as TensorAllocInit<CPU>>::init(&lhs.shape(), &Default::default());

        unsafe {
            interface::cpu_tensor_mul(
                out.into_interface(),
                lhs.into_interface(),
                rhs.into_interface(),
            );
        };

        out
    }
}


impl<Lhs, Rhs, L, R, O> TensorDiv<CPU, Rhs> for Lhs
where
    Lhs: Promote<Rhs> + DTypeImpl<CPU, Prototype = CPUTensorPrototype<L>>,
    Rhs: DTypeImpl<CPU, Prototype = CPUTensorPrototype<R>>,
    Lhs: TensorCast<CPU, <Lhs as Promote<Rhs>>::Output>,
    Rhs: TensorCast<CPU, <Lhs as Promote<Rhs>>::Output>,
    <Lhs as Promote<Rhs>>::Output: DTypeImpl<CPU, Prototype = CPUTensorPrototype<O>>
        + TensorBroadcast<CPU>
        + TensorAllocInit<CPU>,
    CPUTensorPrototype<L>: TensorPrototype<CPU>,
    CPUTensorPrototype<R>: TensorPrototype<CPU>,
    CPUTensorPrototype<O>: TensorPrototype<CPU>,
{
    type Output = <Lhs as Promote<Rhs>>::Output;

    fn div(
        lhs: &Self::Prototype,
        rhs: &Rhs::Prototype,
    ) -> <Self::Output as DTypeImpl<CPU>>::Prototype {

        let (lhs, rhs) = broadcast_prot::<CPU, Self::Output, Self::Output>(
            &<Lhs as TensorCast<CPU, Self::Output>>::cast(lhs),
            &<Rhs as TensorCast<CPU, Self::Output>>::cast(rhs),
        )
        .unwrap_or_else(|| {
            panic!(
                "Two tensors with shapes ({:?}, {:?}) cannot be broadcast.",
                lhs.shape(),
                rhs.shape()
            )
        });

        let out = <Self::Output as TensorAllocInit<CPU>>::init(&lhs.shape(), &Default::default());

        unsafe {
            interface::cpu_tensor_div(
                out.into_interface(),
                lhs.into_interface(),
                rhs.into_interface(),
            );
        };

        out
    }
}
