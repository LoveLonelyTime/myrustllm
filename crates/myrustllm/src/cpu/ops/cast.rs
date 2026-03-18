use crate::common::init::TensorAllocInit;
use crate::common::ops::{
    cast::{TensorCast, TensorCopy, TensorReshape},
    view::{TensorBroadcast, TensorView},
};
use crate::common::{DTypeImpl, DTypeOf, Shape, TensorPrototype};
use crate::cpu::impls::{CPU, CPUTensorPrototype};
use crate::cpu::interface;
use crate::cpu::interface::IntoInterface;

// CPUTensorPrototype<M> <- cast -> CPUTensorPrototype<N>
impl<
    T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<M>> + DTypeOf,
    Dst: DTypeImpl<CPU, Prototype = CPUTensorPrototype<N>> + TensorAllocInit<CPU> + DTypeOf,
    M,
    N,
> TensorCast<CPU, Dst> for T
where
    T::Prototype: TensorPrototype<CPU>,
    Dst::Prototype: TensorPrototype<CPU>,
{
    fn cast(src: &Self::Prototype) -> <Dst as DTypeImpl<CPU>>::Prototype {
        // If the source and destination data types are the same, we can simply transmute the tensor prototype without copying data.
        if <T as DTypeOf>::DTYPE == <Dst as DTypeOf>::DTYPE {
            return unsafe { std::mem::transmute(src.clone()) };
        }

        let out = <Dst as TensorAllocInit<CPU>>::init(&src.shape(), &Default::default());
        unsafe {
            interface::cpu_tensor_cast(out.into_interface(), src.into_interface());
        }
        out
    }
}

impl<
    Dst: DTypeImpl<CPU, Prototype: IntoInterface> + TensorBroadcast<CPU> + DTypeOf,
    Src: DTypeImpl<CPU> + TensorCast<CPU, Dst>,
> TensorCopy<CPU, Src> for Dst
{
    fn copy(dst: &mut Self::Prototype, src: &<Src as DTypeImpl<CPU>>::Prototype) {
        let src = Dst::broadcast_to(&Src::cast(src), &dst.shape()).expect(&format!(
            "Src with shape {:?} cannot broadcast to shape {:?} of dst.",
            src.shape(),
            dst.shape()
        ));

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
