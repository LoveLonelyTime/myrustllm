use crate::common::math::{TensorAddReduce, TensorCopy, TensorMatmul, TensorPermute};
use crate::common::{Shape, Tensor};
use crate::cpu::dynamic::CPUGenericTensor;
use crate::cpu::interface;
use crate::cpu::mem::RawData;
use crate::cpu::tensor::{CPUTensor, broadcast};

// ================================================== COPY ==================================================
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

// ================================================== COPY ==================================================

// ================================================== ADD ==================================================

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

impl std::ops::Add for &CPUGenericTensor {
    type Output = CPUGenericTensor;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (CPUGenericTensor::F32(lhs), CPUGenericTensor::F32(rhs)) => (lhs + rhs).into(),
            _ => panic!("Not supported!"),
        }
    }
}

impl TensorAddBase for f32 {
    fn add(lhs: &CPUTensor<Self>, rhs: &CPUTensor<Self>) -> CPUTensor<Self::Output> {
        let (lhs, rhs) = broadcast(lhs, rhs).expect(&format!(
            "Two tensors with shapes ({:?}, {:?}) cannot be broadcast.",
            lhs.shape(),
            rhs.shape()
        ));

        let out = CPUTensor::alloc(&lhs.shape());

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

// ================================================== ADD ==================================================

// ================================================== ADD_ ==================================================
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

impl std::ops::AddAssign<&CPUGenericTensor> for CPUGenericTensor {
    fn add_assign(&mut self, rhs: &CPUGenericTensor) {
        match (self, rhs) {
            (CPUGenericTensor::F32(lhs), CPUGenericTensor::F32(rhs)) => *lhs += rhs,
            _ => panic!("Not supported!"),
        };
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

// ================================================== ADD_ ==================================================

// ================================================== ADD_R ==================================================

pub trait TensorAddReduceBase: RawData + std::ops::Add<Self, Output: RawData> {
    fn add_reduce(
        tensor: &CPUTensor<Self>,
        dims: Option<&[usize]>,
        keep_dim: bool,
    ) -> CPUTensor<Self::Output>;
}

impl<T: RawData + TensorAddReduceBase> TensorAddReduce for &CPUTensor<T> {
    type Output = CPUTensor<T::Output>;
    fn add_reduce(self, dims: Option<&[usize]>, keep_dim: bool) -> Self::Output {
        T::add_reduce(self, dims, keep_dim)
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
        let res_tensor = CPUTensor::alloc(&old_shape_v.clone().into());

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

// ================================================== ADD_R ==================================================

// ================================================== ADD_ ==================================================
pub trait TensorSubAssignBase<Rhs: RawData = Self>: RawData + std::ops::SubAssign<Rhs> {
    fn sub_assign(lhs: &mut CPUTensor<Self>, rhs: &CPUTensor<Rhs>);
}

impl<T: TensorSubAssignBase<Rhs>, Rhs: RawData> std::ops::SubAssign<&CPUTensor<Rhs>>
    for CPUTensor<T>
{
    fn sub_assign(&mut self, rhs: &CPUTensor<Rhs>) {
        <T as TensorSubAssignBase<Rhs>>::sub_assign(self, rhs);
    }
}

impl std::ops::SubAssign<&CPUGenericTensor> for CPUGenericTensor {
    fn sub_assign(&mut self, rhs: &CPUGenericTensor) {
        match (self, rhs) {
            (CPUGenericTensor::F32(lhs), CPUGenericTensor::F32(rhs)) => *lhs -= rhs,
            _ => panic!("Not supported!"),
        };
    }
}

impl TensorSubAssignBase for f32 {
    fn sub_assign(lhs: &mut CPUTensor<Self>, rhs: &CPUTensor<Self>) {
        let rhs = rhs.broadcast_to(&lhs.shape()).expect(&format!(
            "The rhs tensor with the shape {:?} cannot be broadcast to the shape {:?} of the lhs tensor.",
            rhs.shape(),
            lhs.shape()
        ));

        unsafe {
            interface::cpu_tensor_sub_f32_(lhs.into_interface(), rhs.into_interface());
        };
    }
}

// ================================================== ADD_ ==================================================

// ================================================== MUL ==================================================

pub trait TensorMulBase<Rhs: RawData = Self>:
    RawData + std::ops::Mul<Rhs, Output: RawData>
{
    fn mul(lhs: &CPUTensor<Self>, rhs: &CPUTensor<Rhs>) -> CPUTensor<Self::Output>;
}

impl<T: TensorAddBase<Rhs>, Rhs: RawData> std::ops::Mul<&CPUTensor<Rhs>> for &CPUTensor<T> {
    type Output = CPUTensor<T::Output>;

    fn mul(self, rhs: &CPUTensor<Rhs>) -> Self::Output {
        <T as TensorAddBase<Rhs>>::add(self, rhs)
    }
}

impl std::ops::Mul for &CPUGenericTensor {
    type Output = CPUGenericTensor;

    fn mul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (CPUGenericTensor::F32(lhs), CPUGenericTensor::F32(rhs)) => (lhs * rhs).into(),
            _ => panic!("Not supported!"),
        }
    }
}

impl TensorMulBase for f32 {
    fn mul(lhs: &CPUTensor<Self>, rhs: &CPUTensor<Self>) -> CPUTensor<Self::Output> {
        let (lhs, rhs) = broadcast(lhs, rhs).expect(&format!(
            "Two tensors with shapes ({:?}, {:?}) cannot be broadcast.",
            lhs.shape(),
            rhs.shape()
        ));

        let out = CPUTensor::alloc(&lhs.shape());

        unsafe {
            interface::cpu_tensor_mul_f32(
                out.into_interface(),
                lhs.into_interface(),
                rhs.into_interface(),
            );
        };

        out
    }
}

// ================================================== MUL ==================================================

// ================================================== MATMUL ==================================================
pub trait TensorMatmulBase<Rhs: RawData = Self>:
    RawData + std::ops::Mul<Rhs, Output: RawData + std::ops::Add<Output: RawData>>
{
    fn matmul(
        lhs: &CPUTensor<Self>,
        rhs: &CPUTensor<Rhs>,
    ) -> CPUTensor<<<Self as std::ops::Mul<Rhs>>::Output as std::ops::Add>::Output>;
}

impl<T: TensorMatmulBase<Rhs>, Rhs: RawData> TensorMatmul<&CPUTensor<Rhs>> for &CPUTensor<T> {
    type Output = CPUTensor<<<T as std::ops::Mul<Rhs>>::Output as std::ops::Add>::Output>;

    fn matmul(self, rhs: &CPUTensor<Rhs>) -> Self::Output {
        T::matmul(self, rhs)
    }
}

impl TensorMatmul for &CPUGenericTensor {
    type Output = CPUGenericTensor;

    fn matmul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (CPUGenericTensor::F32(lhs), CPUGenericTensor::F32(rhs)) => (lhs.matmul(rhs)).into(),
            _ => panic!("Not supported!"),
        }
    }
}

impl TensorMatmulBase for f32 {
    fn matmul(
        lhs: &CPUTensor<Self>,
        rhs: &CPUTensor<Self>,
    ) -> CPUTensor<<<Self as std::ops::Mul<Self>>::Output as std::ops::Add>::Output> {
        let (lhs_batch_shape, m, k) = (
            Shape::from(&lhs.shape()[0..lhs.dims() - 2]),
            lhs.shape()[lhs.dims() - 2],
            lhs.shape()[lhs.dims() - 1],
        );
        let (rhs_batch_shape, k2, n) = (
            Shape::from(&rhs.shape()[0..rhs.dims() - 2]),
            rhs.shape()[rhs.dims() - 2],
            rhs.shape()[rhs.dims() - 1],
        );
        assert!(k == k2);

        let batch_shape =
            Shape::broadcast_shape(&lhs_batch_shape, &rhs_batch_shape).expect(&format!(
                "Lhs with shape {:?} and rhs with shape {:?} cannot be broadcast.",
                lhs_batch_shape, rhs_batch_shape
            ));

        let mut lhs_shape_v = Vec::from(batch_shape.as_ref());
        lhs_shape_v.extend(&[m, k]);
        let lhs = lhs.broadcast_to(&Shape::from(lhs_shape_v)).unwrap();

        let mut rhs_shape_v = Vec::from(batch_shape.as_ref());
        rhs_shape_v.extend(&[k, n]);
        let rhs = rhs.broadcast_to(&Shape::from(rhs_shape_v)).unwrap();

        let mut out_shape_v = Vec::from(batch_shape.as_ref());
        out_shape_v.extend(&[m, n]);
        let out = CPUTensor::alloc(&Shape::from(out_shape_v));

        unsafe {
            interface::cpu_tensor_matmul_f32(
                out.into_interface(),
                lhs.into_interface(),
                rhs.into_interface(),
            );
        }

        out
    }
}

// ================================================== MATMUL ==================================================

// ================================================== PERMUTE ==================================================

impl<T: RawData> TensorPermute for CPUTensor<T> {
    fn permute(&self, dims: &[usize]) -> Self {
        // Check
        assert!(
            dims.len() == self.dims(),
            "The size of dims must be {}, but got {}.",
            self.dims(),
            dims.len()
        );

        let mut checklist = vec![false; self.dims()];
        for &i in dims {
            assert!(
                i < self.dims(),
                "Index {} is out of bounds of the dimensions with size {}.",
                i,
                self.dims()
            );
            checklist[i] = true;
        }
        assert!(checklist.iter().all(|&x| x), "Invalid dims: {:?}.", dims);

        // Permute
        let mut new_shape_v = vec![0; self.dims()];
        let mut new_stride_v = vec![0; self.dims()];
        for (new_i, &i) in dims.iter().enumerate() {
            new_shape_v[new_i] = self.shape()[i];
            new_stride_v[new_i] = self.stride()[i];
        }

        CPUTensor::new(
            self.data().clone(),
            &new_shape_v.into(),
            &new_stride_v.into(),
            self.offset(),
        )
    }
}

impl TensorPermute for CPUGenericTensor {
    fn permute(&self, dims: &[usize]) -> Self {
        match self {
            CPUGenericTensor::F32(t) => t.permute(dims).into(),
            _ => panic!("Not supported!"),
        }
    }
}

// ================================================== PERMUTE ==================================================
