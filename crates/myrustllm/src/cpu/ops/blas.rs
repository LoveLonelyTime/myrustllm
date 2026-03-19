use crate::common::init::TensorAllocInit;
use crate::common::ops::{
    blas::TensorMatmul, cast::TensorCast, promote::Promote, view::TensorBroadcast,
};
use crate::common::shape::broadcast_shape;
use crate::common::{DTypeImpl, Shape, TensorPrototype};
use crate::cpu::impls::{CPU, CPUTensorPrototype};
use crate::cpu::interface;
use crate::cpu::interface::IntoInterface;

impl<Lhs, Rhs, L, R, O> TensorMatmul<CPU, Rhs> for Lhs
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

    fn matmul(
        lhs: &Self::Prototype,
        rhs: &<Rhs as DTypeImpl<CPU>>::Prototype,
    ) -> <Self::Output as DTypeImpl<CPU>>::Prototype {
        // Cast
        let (lhs, rhs) = (
            <Lhs as TensorCast<CPU, Self::Output>>::cast(lhs),
            <Rhs as TensorCast<CPU, Self::Output>>::cast(rhs),
        );

        // Broadcast
        let (lhs_batch_shape, m, k) = (
            Shape::from(lhs.shape()[0..lhs.dims() - 2].to_vec()),
            lhs.shape()[lhs.dims() - 2],
            lhs.shape()[lhs.dims() - 1],
        );
        let (rhs_batch_shape, k2, n) = (
            Shape::from(rhs.shape()[0..rhs.dims() - 2].to_vec()),
            rhs.shape()[rhs.dims() - 2],
            rhs.shape()[rhs.dims() - 1],
        );
        assert!(k == k2);

        let batch_shape = broadcast_shape(&lhs_batch_shape, &rhs_batch_shape).expect(&format!(
            "Lhs with shape {:?} and rhs with shape {:?} cannot be broadcast.",
            lhs_batch_shape, rhs_batch_shape
        ));

        let mut lhs_shape_v = Vec::from(batch_shape.as_ref());
        lhs_shape_v.extend(&[m, k]);

        let lhs = Self::Output::broadcast_to(&lhs, &Shape::from(lhs_shape_v)).unwrap();

        let mut rhs_shape_v = Vec::from(batch_shape.as_ref());
        rhs_shape_v.extend(&[k, n]);
        let rhs = Self::Output::broadcast_to(&rhs, &Shape::from(rhs_shape_v)).unwrap();

        let mut out_shape_v = Vec::from(batch_shape.as_ref());
        out_shape_v.extend(&[m, n]);
        let out = Self::Output::init(&Shape::from(out_shape_v), &Default::default());

        unsafe {
            interface::cpu_tensor_matmul(
                out.into_interface(),
                lhs.into_interface(),
                rhs.into_interface(),
            );
        }

        out
    }
}
