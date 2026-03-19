use crate::{
    autograd::impls::{Autograd, AutogradPrototype},
    common::{
        DTypeImpl, Impl,
        init::TensorZerosInit,
        ops::{binary_ops::TensorAdd, reduce_ops::TensorAddReduce},
    },
};

impl<
    I: Impl,
    TI: DTypeImpl<I> + TensorAddReduce<I>,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
> TensorAddReduce<Autograd<I, GI>> for TI
{
    fn add_reduce(
        tensor: &Self::Prototype,
        dims: Option<&[usize]>,
        keep_dim: bool,
    ) -> Self::Prototype {
        // TODO

        let output = tensor.tensor().add_reduce(dims, keep_dim);

        AutogradPrototype::leaf(output)
    }
}
