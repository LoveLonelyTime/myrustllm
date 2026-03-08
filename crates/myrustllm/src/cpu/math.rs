use crate::common::math::{TensorAddReduce, TensorCopy};
use crate::cpu::dynamic::CPUGenericTensor;
use crate::cpu::interface;
use crate::cpu::mem::RawData;
use crate::cpu::tensor::{CPUTensor, broadcast};
use crate::common::Tensor;

pub trait TensorCopyBase<Rhs: RawData = Self>: RawData {
    fn copy_from(lhs: &mut CPUTensor<Self>, rhs: &CPUTensor<Rhs>);
}

impl<T: TensorCopyBase<Rhs>, Rhs: RawData> TensorCopy<&CPUTensor<Rhs>> for CPUTensor<T> {
    fn copy_from(&mut self, rhs: &CPUTensor<Rhs>) {
        T::copy_from(self, rhs);
    }
}

impl TensorCopyBase for f32 {
    fn copy_from(lhs: &mut CPUTensor<Self>, rhs: &CPUTensor<Self>) {
        let rhs = rhs.broadcast_to(&lhs.shape()).expect(&format!(
            "Rhs with shape {:?} cannot broadcast to shape {:?} of lhs.",
            rhs.shape(),
            lhs.shape()
        ));

        unsafe {
            interface::cpu_tensor_copy_f32(lhs.into_interface(), rhs.into_interface());
        }
    }
}

pub trait TensorAddBase<Rhs: RawData = Self>:
    RawData + std::ops::Add<Rhs, Output: RawData>
{
    fn add(lhs: &CPUTensor<Self>, rhs: &CPUTensor<Rhs>) -> CPUTensor<Self::Output>;
}

impl<T: TensorAddBase<Rhs>, Rhs: RawData> std::ops::Add<&CPUTensor<Rhs>> for &CPUTensor<T> {
    type Output = CPUTensor<T::Output>;

    fn add(self, rhs: &CPUTensor<Rhs>) -> Self::Output {
        <T as TensorAddBase<Rhs>>::add(self, rhs)
    }
}

impl TensorAddBase for f32 {
    fn add(lhs: &CPUTensor<Self>, rhs: &CPUTensor<Self>) -> CPUTensor<Self::Output> {
        let (lhs, rhs) = broadcast(lhs, rhs).expect(&format!(
            "Two tensors with shapes ({:?}, {:?}) cannot be broadcast.",
            lhs.shape(),
            rhs.shape()
        ));

        let out = CPUTensor::from_shape(&lhs.shape());

        unsafe {
            interface::cpu_tensor_add_f32(
                out.into_interface(),
                lhs.into_interface(),
                rhs.into_interface(),
            );
        };

        out
    }
}

pub trait TensorAddAssignBase<Rhs: RawData = Self>: RawData + std::ops::AddAssign<Rhs> {
    fn add_assign(lhs: &mut CPUTensor<Self>, rhs: &CPUTensor<Rhs>);
}

impl<T: TensorAddAssignBase<Rhs>, Rhs: RawData> std::ops::AddAssign<&CPUTensor<Rhs>>
    for CPUTensor<T>
{
    fn add_assign(&mut self, rhs: &CPUTensor<Rhs>) {
        <T as TensorAddAssignBase<Rhs>>::add_assign(self, rhs);
    }
}

impl TensorAddAssignBase for f32 {
    fn add_assign(lhs: &mut CPUTensor<Self>, rhs: &CPUTensor<Self>) {
        let rhs = rhs.broadcast_to(&lhs.shape()).expect(&format!(
            "The rhs tensor with the shape {:?} cannot be broadcast to the shape {:?} of the lhs tensor.",
            rhs.shape(),
            lhs.shape()
        ));

        unsafe {
            interface::cpu_tensor_add_f32_(lhs.into_interface(), rhs.into_interface());
        };
    }
}

pub trait TensorAddReduceBase: RawData + std::ops::Add<Self, Output: RawData> {
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
        let reduce_tensor = tensor.permute(&permute_list);
        let res_tensor = CPUTensor::from_shape(&old_shape_v.clone().into());

        unsafe {
            interface::cpu_tensor_add_f32_r(
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

        let out = res_tensor.reshape(&old_shape_v.into());

        if keep_dim {
            out.permute(&permute_list_rev)
        } else {
            out
        }
    }
}


impl <T: RawData + TensorAddReduceBase> TensorAddReduce for &CPUTensor<T> {
    type Output = CPUTensor<T::Output>;
    fn add_reduce(self, dims: Option<&[usize]>, keep_dim: bool) -> Self::Output {
        T::add_reduce(self, dims, keep_dim)
    }
}

impl std::ops::Add for &CPUGenericTensor {
    type Output = CPUGenericTensor;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (CPUGenericTensor::F32(lhs), CPUGenericTensor::F32(rhs)) => {
                (lhs + rhs).into()
            }
            _ => panic!("Not supported!"),
        }
    }
}

impl std::ops::AddAssign<&CPUGenericTensor> for CPUGenericTensor {
    fn add_assign(&mut self, rhs: &CPUGenericTensor) {
        match (self, rhs) {
            (CPUGenericTensor::F32(lhs), CPUGenericTensor::F32(rhs)) => *lhs += rhs,
            _ => panic!("Not supported!"),
        };
    }
}

impl TensorAddReduce for &CPUGenericTensor {
    type Output = CPUGenericTensor;
    fn add_reduce(self, dims: Option<&[usize]>, keep_dim: bool) -> Self::Output {
        match self {
            CPUGenericTensor::F32(t) => t.add_reduce(dims, keep_dim).into(),
            _ => panic!("Not supported!"),
        }
    }
}
