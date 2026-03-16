use crate::{
    autograd::init,
    common::{
        Shape, Tensor,
        dtype::{Any, DTypeImpl},
        init::TensorAllocInit,
        ops::{
            cast::{TensorCast, TensorCopy, TensorReshape},
            promote::Promote,
        },
        tensor::TensorPrototype,
    },
    cpu::{
        impls::{CPU, CPUTensorAnyPrototype, CPUTensorPrototype},
        interface::{self, IntoInterface},
    },
};

use crate::common::ops::view::TensorBroadcast;
use crate::common::ops::view::TensorView;

impl<
    T: DTypeImpl<CPU, Prototype: IntoInterface>,
    Dst: DTypeImpl<CPU, Prototype: IntoInterface> + TensorAllocInit<CPU>,
> TensorCast<CPU, Dst> for T
{
    fn cast(src: &Self::Prototype) -> <Dst as DTypeImpl<CPU>>::Prototype {
        let out = <Dst as TensorAllocInit<CPU>>::init(&src.shape(), &());
        unsafe {
            interface::cpu_tensor_cast(out.into_interface(), src.into_interface());
        }
        out
    }
}

// impl<Dst: DTypeImpl<CPU, Prototype: IntoInterface> + TensorAllocInit<CPU>> TensorCastImpl<CPU, Dst>
//     for Any
// {
//     fn cast(src: &Self::Prototype) -> <Dst as DTypeImpl<CPU>>::Prototype {
//         let out = <Dst as TensorAllocInit<CPU>>::init(&src.shape(), &());
//         unsafe {
//             interface::cpu_tensor_cast(out.into_interface(), src.into_interface());
//         }
//         out
//     }
// }

impl<Src: DTypeImpl<CPU, Prototype: Into<<Any as DTypeImpl<CPU>>::Prototype> + Clone>>
    TensorCast<CPU, Any> for Src
{
    fn cast(src: &Self::Prototype) -> <Any as DTypeImpl<CPU>>::Prototype {
        src.clone().into()
    }
}

impl<
    Dst: DTypeImpl<CPU, Prototype: IntoInterface>,
    Src: DTypeImpl<CPU, Prototype: IntoInterface> + TensorBroadcast<CPU> + TensorCast<CPU, Dst>,
> TensorCopy<CPU, Src> for Dst
{
    fn copy(dst: &mut Self::Prototype, src: &<Src as DTypeImpl<CPU>>::Prototype) {
        let src = Src::cast(&Src::broadcast_to(src, &dst.shape()).expect(&format!(
            "Src with shape {:?} cannot broadcast to shape {:?} of dst.",
            src.shape(),
            dst.shape()
        )));

        unsafe {
            interface::cpu_tensor_copy(dst.into_interface(), src.into_interface());
        }
    }
}

impl<Src: DTypeImpl<CPU> + TensorView<CPU> + TensorAllocInit<CPU> + TensorCopy<CPU, Src>>
    TensorReshape<CPU> for Src
{
    fn reshape(src: &Self::Prototype, new_shape: &Shape) -> Self::Prototype {
        // Try view
        if let Some(t) = Src::view(src, new_shape) {
            return t;
        }

        let mut new_shape_v: Vec<usize> = new_shape.as_ref().into();

        // Check inferred dim
        let mut inferred_dim = None;
        let mut known_size = 1;
        for (i, &dim) in new_shape_v.iter().enumerate() {
            if dim == 0 {
                assert!(
                    inferred_dim.is_none(),
                    "The number of auto-inferred dims is greater than 1."
                );

                inferred_dim = Some(i);
            } else {
                known_size *= dim;
            }
        }

        if let Some(i) = inferred_dim {
            assert!(
                src.shape().numel() % known_size == 0,
                "Cannot infer the auto-inferred dim."
            );

            new_shape_v[i] = src.shape().numel() / known_size;
            known_size *= new_shape_v[i];
        }

        assert!(
            src.shape().numel() == known_size,
            "The new number {} of elements is not equal to the original number {}.",
            known_size,
            src.shape().numel()
        );

        // Copy
        let mut out = <Src as TensorAllocInit<CPU>>::init(&new_shape_v.into(), &());
        <Src as TensorCopy<CPU, Src>>::copy(&mut out, src);

        out
    }
}
