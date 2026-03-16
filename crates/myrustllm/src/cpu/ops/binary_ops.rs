use crate::{
    common::{
        dtype::DTypeImpl,
        init::TensorAllocInit,
        ops::{
            binary_ops::TensorAdd,
            cast::TensorCast,
            promote::Promote,
            view::{TensorBroadcast, broadcast_prot},
        },
        tensor::TensorPrototype,
    },
    cpu::{
        impls::CPU,
        interface::{self, IntoInterface},
    },
};

impl<
    Lhs: DTypeImpl<CPU, Prototype: IntoInterface>
        + Promote<
            Rhs,
            Output: DTypeImpl<CPU, Prototype: IntoInterface>
                        + TensorBroadcast<CPU>
                        + TensorAllocInit<CPU>,
        >,
    Rhs: DTypeImpl<CPU, Prototype: IntoInterface>,
> TensorAdd<CPU, Rhs> for Lhs
{
    type Output = <Lhs as Promote<Rhs>>::Output;

    fn add(
        lhs: &Self::Prototype,
        rhs: &<Rhs as DTypeImpl<CPU>>::Prototype,
    ) -> <Self::Output as DTypeImpl<CPU>>::Prototype {
        let (lhs, rhs) = broadcast_prot::<CPU, Self::Output, Self::Output>(
            &<Lhs as TensorCast<CPU, Self::Output>>::cast(lhs),
            &<Rhs as TensorCast<CPU, Self::Output>>::cast(rhs),
        )
        .expect(&format!(
            "Two tensors with shapes ({:?}, {:?}) cannot be broadcast.",
            lhs.shape(),
            rhs.shape()
        ));

        let out = <Self::Output as TensorAllocInit<CPU>>::init(&lhs.shape(), &());

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
