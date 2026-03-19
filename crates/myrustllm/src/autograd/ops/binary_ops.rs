use crate::{
    autograd::{graph::GraphNode, impls::Autograd},
    common::{
        Shape, Tensor, TensorPrototype,
        dtype::DTypeImpl,
        impls::Impl,
        init::TensorZerosInit,
        ops::{binary_ops::TensorAdd, reduce_ops::TensorAddReduce},
    },
};

impl<
    I: Impl,
    GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
    TI: DTypeImpl<I> + TensorAdd<I, Rhs>,
    Rhs: DTypeImpl<I>,
> TensorAdd<Autograd<I, GI>, Rhs> for TI
{
    type Output = TI::Output;
    fn add(
        lhs: &Self::Prototype,
        rhs: &<Rhs as DTypeImpl<Autograd<I, GI>>>::Prototype,
    ) -> <Self::Output as DTypeImpl<Autograd<I, GI>>>::Prototype {
        fn reduce_grad<
            I: Impl,
            GI: DTypeImpl<I> + TensorZerosInit<I> + TensorAdd<I, GI, Output = GI> + TensorAddReduce<I>,
        >(
            grad: &Tensor<Autograd<I, GI>, GI>,
            target_shape: &Shape,
        ) -> Tensor<Autograd<I, GI>, GI> {
            let mut reduce_dims = Vec::new();
            let diff = grad.dims() - target_shape.len();

            for dim in 0..diff {
                reduce_dims.push(dim);
            }

            let grad = grad.add_reduce(Some(&reduce_dims), false);
            reduce_dims.clear();

            for dim in 0..target_shape.len() {
                if target_shape[dim] == 1 && grad.shape()[dim] != 1 {
                    reduce_dims.push(dim);
                }
            }

            grad.add_reduce(Some(&reduce_dims), true)
        }

        let lhs_shape = lhs.shape();
        let rhs_shape = rhs.shape();
        let output = lhs.tensor() + rhs.tensor();
        let backward_fn = move |grad_inputs: &[Tensor<Autograd<I, GI>, GI>]| {
            vec![
                reduce_grad(&grad_inputs[0], &lhs_shape),
                reduce_grad(&grad_inputs[0], &rhs_shape),
            ]
        };

        GraphNode::new()
            .add_input(lhs)
            .add_input(rhs)
            .set_backward_op(backward_fn)
            .attch_output(output, 0)
    }
}
