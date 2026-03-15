use crate::{
    common::{
        dtype::{DTypeImpl, StdDType},
        init::TensorAllocInit,
        ops::{
            binary_ops::TensorAddImpl, cast::TensorCastImpl, promote::Promote, view::broadcast_prot,
        },
        tensor::TensorPrototype,
    },
    cpu::{
        impls::CPU,
        interface::{self, IntoInterface},
    },
};

impl<Lhs: StdDType + Promote<Rhs, Output: StdDType>, Rhs: StdDType> TensorAddImpl<CPU, Rhs>
    for Lhs
{
    type Output = <Lhs as Promote<Rhs>>::Output;
    fn add(
        lhs: &Self::Prototype,
        rhs: &<Rhs as DTypeImpl<CPU>>::Prototype,
    ) -> <Self::Output as DTypeImpl<CPU>>::Prototype {
        let (lhs, rhs) = broadcast_prot::<CPU, Self::Output, Self::Output>(
            &<Lhs as TensorCastImpl<CPU, Self::Output>>::cast(lhs),
            &<Rhs as TensorCastImpl<CPU, Self::Output>>::cast(rhs),
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
