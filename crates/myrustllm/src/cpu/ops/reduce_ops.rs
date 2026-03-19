use crate::common::init::TensorAllocInit;
use crate::common::ops::{cast::TensorReshape, reduce_ops::TensorAddReduce, view::TensorPermute};
use crate::common::{DTypeImpl, TensorPrototype};
use crate::cpu::impls::{CPU, CPUTensorPrototype};
use crate::cpu::interface;
use crate::cpu::interface::IntoInterface;

impl<T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<U>>, U> TensorAddReduce<CPU> for T
where
    CPUTensorPrototype<U>: TensorPrototype<CPU>,
    T: TensorAllocInit<CPU> + TensorPermute<CPU> + TensorReshape<CPU>,
{
    fn add_reduce(
        tensor: &Self::Prototype,
        dims: Option<&[usize]>,
        keep_dim: bool,
    ) -> Self::Prototype {
        let mut reduce_dims = Vec::new();
        if let Some(dims) = dims {
            reduce_dims.extend(dims);
        } else {
            reduce_dims.extend(0..tensor.dims());
        }

        // Cut
        let mut permute_list = Vec::new();
        let mut old_shape_v = Vec::new();
        for i in 0..tensor.dims() {
            if !reduce_dims.contains(&i) {
                permute_list.push(i);
                old_shape_v.push(tensor.shape()[i]);
            }
        }
        permute_list.extend(&reduce_dims);

        let mut permute_list_rev = vec![0; tensor.dims()];
        for (i, &dim) in permute_list.iter().enumerate() {
            permute_list_rev[dim] = i;
        }

        // Reduce
        let reduce_tensor = T::permute(tensor, &permute_list);

        let res_tensor = T::init(&old_shape_v.clone().into(), &Default::default());

        unsafe {
            interface::cpu_tensor_reduce_add(
                res_tensor.into_interface(),
                reduce_tensor.into_interface(),
                tensor.dims() - reduce_dims.len(),
            );
        };

        // Expand
        if keep_dim {
            for _ in 0..reduce_dims.len() {
                old_shape_v.push(1);
            }
        }

        let out = T::reshape(&res_tensor, &old_shape_v.into());

        if keep_dim {
            T::permute(&out, &permute_list_rev)
        } else {
            out
        }
    }
}
