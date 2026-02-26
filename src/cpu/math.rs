use crate::cpu::interface;
use crate::cpu::tensor::{CPUTensor, Tensor, broadcast};

pub trait TensorCopyBase<Rhs: Copy = Self>: Copy {
    fn copy_from(lhs: &mut CPUTensor<Self>, rhs: &CPUTensor<Rhs>);
}

pub trait TensorCopy<Rhs = Self> {
    fn copy_from(&mut self, rhs: &CPUTensor<Rhs>);
}

impl<T: TensorCopyBase<Rhs>, Rhs: Copy> TensorCopy<Rhs> for CPUTensor<T> {
    fn copy_from(&mut self, rhs: &CPUTensor<Rhs>) {
        T::copy_from(self, rhs);
    }
}

impl TensorCopyBase for f32 {
    fn copy_from(lhs: &mut CPUTensor<Self>, rhs: &CPUTensor<Self>) {
        let rhs = rhs.broadcast_to(&lhs.shape()).expect(&format!(
            "Rhs with shape {} cannot broadcast to shape {} of lhs.",
            rhs.shape(),
            lhs.shape()
        ));

        unsafe {
            interface::cpu_tensor_copy_f32((lhs as &CPUTensor<Self>).into(), (&rhs).into());
        }
    }
}

pub trait TensorAddBase<Rhs = Self>: std::ops::Add<Rhs> + Sized {
    fn add(lhs: &CPUTensor<Self>, rhs: &CPUTensor<Rhs>) -> CPUTensor<Self::Output>;
}

impl<T: TensorAddBase<Rhs>, Rhs> std::ops::Add<&CPUTensor<Rhs>> for &CPUTensor<T> {
    type Output = CPUTensor<T::Output>;

    fn add(self, rhs: &CPUTensor<Rhs>) -> Self::Output {
        <T as TensorAddBase<Rhs>>::add(self, rhs)
    }
}

impl TensorAddBase for f32 {
    fn add(lhs: &CPUTensor<Self>, rhs: &CPUTensor<Self>) -> CPUTensor<Self::Output> {
        let (lhs, rhs) = broadcast(lhs, rhs).expect(&format!(
            "Two tensors with shapes ({}, {}) cannot be broadcast.",
            lhs.shape(),
            rhs.shape()
        ));

        let out = CPUTensor::from_shape(&lhs.shape());

        unsafe {
            interface::cpu_tensor_add_f32((&out).into(), (&lhs).into(), (&rhs).into());
        };

        out
    }
}

pub trait TensorAddAssignBase<Rhs = Self>: std::ops::AddAssign<Rhs> + Sized {
    fn add_assign(lhs: &mut CPUTensor<Self>, rhs: &CPUTensor<Rhs>);
}

impl<T: TensorAddAssignBase<Rhs>, Rhs> std::ops::AddAssign<&CPUTensor<Rhs>> for CPUTensor<T> {
    fn add_assign(&mut self, rhs: &CPUTensor<Rhs>) {
        <T as TensorAddAssignBase<Rhs>>::add_assign(self, rhs);
    }
}

impl TensorAddAssignBase for f32 {
    fn add_assign(lhs: &mut CPUTensor<Self>, rhs: &CPUTensor<Self>) {
        let rhs = rhs.broadcast_to(&lhs.shape()).expect(&format!(
            "The rhs tensor with the shape {} cannot be broadcast to the shape {} of the lhs tensor.",
            rhs.shape(),
            lhs.shape()
        ));

        unsafe {
            interface::cpu_tensor_add_f32_((lhs as &CPUTensor<Self>).into(), (&rhs).into());
        };
    }
}

pub trait TensorAddReduceBase: std::ops::Add<Self> + Sized {
    fn add_reduce(
        tensor: &CPUTensor<Self>,
        dims: Option<&[usize]>,
        keep_dim: bool,
    ) -> CPUTensor<Self::Output>;
}

impl TensorAddReduceBase for f32 {
    fn add_reduce(
        tensor: &CPUTensor<Self>,
        dims: Option<&[usize]>,
        keep_dim: bool,
    ) -> CPUTensor<Self::Output> {
        let mut batch_size = 1;
        let mut reduce_size = 1;

        let mut reduce_dims = Vec::new();
        if let Some(dims) = dims {
            reduce_dims.extend(dims);
        } else {
            reduce_dims.extend(0..self.dims());
        }

        // Cut
        let mut permute_list = Vec::new();
        let mut old_shape_v = Vec::new();
        for i in 0..self.dims() {
            if reduce_dims.contains(&i) {
                reduce_size *= self.shape()[i];
            } else {
                batch_size *= self.shape()[i];
                permute_list.push(i);
                old_shape_v.push(self.shape()[i]);
            }
        }
        permute_list.extend(&reduce_dims);

        let mut permute_list_rev = vec![0; self.dims()];
        for (i, &dim) in permute_list.iter().enumerate() {
            permute_list_rev[dim] = i;
        }

        // Reduce
        let reduce_tensor = self
            .permute(&permute_list)
            .reshape(&Shape::new(vec![batch_size, reduce_size]));
        let res_tensor = CPUTensor::from_shape(&Shape::new(vec![batch_size]));

        unsafe {
            let reduce_tensor_ptr = reduce_tensor.borrow().as_ptr() as *const libc::c_float;
            let res_tensor_ptr = res_tensor.borrow_mut().as_mut_ptr() as *mut libc::c_float;

            interface::cpu_tensor_add_f32_r(
                reduce_tensor_ptr.add(reduce_tensor.offset()),
                res_tensor_ptr.add(res_tensor.offset()),
                reduce_tensor.stride().as_ptr(),
                res_tensor.stride().as_ptr(),
                reduce_tensor.shape().as_ptr(),
            );
        }

        // Expand
        if keep_dim {
            for _ in 0..reduce_dims.len() {
                old_shape_v.push(1);
            }
        }

        let out = res_tensor.reshape(&Shape::new(old_shape_v));

        if keep_dim {
            out.permute(&permute_list_rev)
        } else {
            out
        }
    }
}

// impl std::ops::Add for &CPUGenericTensor {
//     type Output = CPUGenericTensor;

//     fn add(self, rhs: Self) -> Self::Output {
//         match (self, rhs) {
//             (CPUGenericTensor::F32(lhs), CPUGenericTensor::F32(rhs)) => {
//                 CPUGenericTensor::F32(lhs + rhs)
//             }
//             _ => panic!("Not supported!"),
//         }
//     }
// }

// impl std::ops::AddAssign<&CPUGenericTensor> for CPUGenericTensor {
//     fn add_assign(&mut self, rhs: &CPUGenericTensor) {
//         match (self, rhs) {
//             (CPUGenericTensor::F32(lhs), CPUGenericTensor::F32(rhs)) => *lhs += rhs,
//             _ => panic!("Not supported!"),
//         };
//     }
// }

// pub struct AddGrad {
//     a_shape: Option<Shape>,
//     b_shape: Option<Shape>,
// }

// impl AddGrad {
//     pub fn reduce_grad(grad: &CPUGenericTensor, target_shape: &Shape) -> CPUGenericTensor {
//         todo!()
//     }
// }

// impl CPUOpGrad for AddGrad {
//     fn forward(&mut self, inputs: &[CPUGenericTensor]) -> CPUGenericTensor {
//         assert_eq!(
//             inputs.len(),
//             2,
//             "AddGrad requires two operands, but got {}.",
//             inputs.len()
//         );

//         self.a_shape = Some(inputs[0].shape());
//         self.b_shape = Some(inputs[1].shape());
//         &inputs[0] + &inputs[1]
//     }

//     fn backward(&mut self, grad_inputs: &CPUGenericTensor) -> Vec<CPUGenericTensor> {
//         let a_shape = self
//             .a_shape
//             .take()
//             .expect("AddGrad forward hasn't been called.");
//         let b_shape = self
//             .b_shape
//             .take()
//             .expect("AddGrad forward hasn't been called.");

//         return vec![
//             AddGrad::reduce_grad(grad_inputs, &a_shape),
//             AddGrad::reduce_grad(grad_inputs, &b_shape),
//         ];
//     }
// }

// impl std::ops::Add for &CPUGraphNode {
//     type Output = CPUGraphNode;

//     fn add(self, rhs: Self) -> Self::Output {
//         todo!()
//     }
// }
