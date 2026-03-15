use crate::{
    common::{
        Shape, Tensor,
        dtype::{Any, DTypeImpl, StdDType, StdRType},
        ops::{cast::{TensorCastImpl, TensorCopyImpl, TensorReshapeImpl}, promote::Promote},
        tensor::TensorPrototype,
    },
    cpu::{
        impls::{CPU, CPUTensorAnyPrototype},
        interface::{self, IntoInterface},
    },
};

use crate::common::ops::view::TensorBroadcastImpl;
use crate::common::ops::view::TensorViewImpl;

impl<T: StdDType, Dst: StdDType> TensorCastImpl<CPU, Dst> for T {
    fn cast(src: &Self::Prototype) -> <Dst as DTypeImpl<CPU>>::Prototype {
        let out = <Dst as DTypeImpl<CPU>>::Prototype::alloc(&src.shape());
        unsafe {
            interface::cpu_tensor_cast(out.into_interface(), src.into_interface());
        }
        out
    }
}

impl<Dst: StdDType> TensorCastImpl<CPU, Dst> for Any {
    fn cast(src: &Self::Prototype) -> <Dst as DTypeImpl<CPU>>::Prototype {
        let out = <Dst as DTypeImpl<CPU>>::Prototype::alloc(&src.shape());
        unsafe {
            interface::cpu_tensor_cast(out.into_interface(), src.into_interface());
        }
        out
    }
}

impl<Src: StdDType> TensorCastImpl<CPU, Any> for Src {
    fn cast(src: &Self::Prototype) -> <Any as DTypeImpl<CPU>>::Prototype {
        <Any as DTypeImpl<CPU>>::Prototype::from(src.clone())
    }
}

impl<Dst: StdDType, Src: StdDType> TensorCopyImpl<CPU, Src> for Dst {
    fn copy(dst: &mut Self::Prototype, src: &<Src as DTypeImpl<CPU>>::Prototype) {
        let src = Dst::broadcast_to(&<Src as TensorCastImpl<CPU, Dst>>::cast(src), &dst.shape())
            .expect(&format!(
                "Src with shape {:?} cannot broadcast to shape {:?} of dst.",
                src.shape(),
                dst.shape()
            ));

        unsafe {
            interface::cpu_tensor_copy(dst.into_interface(), src.into_interface());
        }
    }
}

impl<Src: StdDType> TensorReshapeImpl<CPU> for Src {
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
        let mut out = Self::Prototype::alloc(&new_shape_v.into());
        <Src as TensorCopyImpl<CPU, Src>>::copy(&mut out, src);

        out
    }
}
