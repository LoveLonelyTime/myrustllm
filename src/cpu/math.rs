use std::process::Output;

use crate::cpu::autograd::{CPUGraphNode, CPUGraphNodeBase, CPUOpGrad};
use crate::cpu::dynamic::CPUGenericTensor;
use crate::cpu::interface;
use crate::cpu::mem::RawData;
use crate::cpu::shape::Shape;
use crate::cpu::tensor::{CPUTensor, Tensor, broadcast};

pub trait TensorCopyBase<Rhs: RawData = Self>: RawData {
    fn copy_from(lhs: &mut CPUTensor<Self>, rhs: &CPUTensor<Rhs>);
}

pub trait TensorCopy<Rhs = Self> {
    fn copy_from(&mut self, rhs: Rhs);
}

impl<T: TensorCopyBase<Rhs>, Rhs: RawData> TensorCopy<&CPUTensor<Rhs>> for CPUTensor<T> {
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
            "Two tensors with shapes ({}, {}) cannot be broadcast.",
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
            "The rhs tensor with the shape {} cannot be broadcast to the shape {} of the lhs tensor.",
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
        let res_tensor = CPUTensor::from_shape(&Shape::new(old_shape_v.clone()));

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

        let out = res_tensor.reshape(&Shape::new(old_shape_v));

        if keep_dim {
            out.permute(&permute_list_rev)
        } else {
            out
        }
    }
}

pub trait TensorAddReduce{
    type Output;
    fn add_reduce(self, dims: Option<&[usize]>, keep_dim: bool) -> Self::Output;
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


pub struct AddGrad {
    a_shape: Option<Shape>,
    b_shape: Option<Shape>,
}

impl AddGrad {
    pub fn reduce_grad(grad: &CPUGenericTensor, target_shape: &Shape) -> CPUGenericTensor {
        let mut reduce_dims = Vec::new();
        let diff = grad.dims() - target_shape.len();

        for dim in 0..diff {
            reduce_dims.push(dim);
        }

        let grad = grad.add_reduce(Some(&reduce_dims), false);
        reduce_dims.clear();

        for dim in 0..target_shape.len(){
            if target_shape[dim] == 1 && grad.shape()[dim] != 1 {
                reduce_dims.push(dim);
            }
        }

        grad.add_reduce(Some(&reduce_dims), true)
    }
}

impl CPUOpGrad for AddGrad {
    fn forward(&mut self, inputs: &[CPUGenericTensor]) -> CPUGenericTensor {
        assert_eq!(
            inputs.len(),
            2,
            "AddGrad requires two operands, but got {}.",
            inputs.len()
        );

        self.a_shape = Some(inputs[0].shape());
        self.b_shape = Some(inputs[1].shape());
        &inputs[0] + &inputs[1]
    }

    fn backward(&mut self, grad_inputs: &CPUGenericTensor) -> Vec<CPUGenericTensor> {
        let a_shape = self
            .a_shape
            .take()
            .expect("AddGrad forward hasn't been called.");
        let b_shape = self
            .b_shape
            .take()
            .expect("AddGrad forward hasn't been called.");

        return vec![
            AddGrad::reduce_grad(grad_inputs, &a_shape),
            AddGrad::reduce_grad(grad_inputs, &b_shape),
        ];
    }
}

impl std::ops::Add for &CPUGraphNode {
    type Output = CPUGraphNode;

    fn add(self, rhs: &CPUGraphNode) -> Self::Output {
        let mut add_grad = AddGrad {
            a_shape: None,
            b_shape: None
        };
        let a = self.borrow().tensor.clone();
        let b = rhs.borrow().tensor.clone();
        let res = add_grad.forward(&[a, b]);
        CPUGraphNode::new(CPUGraphNodeBase::new(res, vec![self.clone(), rhs.clone()], Some(Box::new(add_grad)), Some(Box::new(|t,g| { 
            println!("Update")
        }) )))
    }
}
