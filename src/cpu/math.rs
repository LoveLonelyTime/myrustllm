use crate::cpu::autograd::{CPUGraphNode, CPUOpGrad};
use crate::cpu::interface;
use crate::cpu::shape::Shape;
use crate::cpu::tensor::{CPUGenericTensor, CPUTensor, Tensor, broadcast};

impl CPUTensor<f32> {
    pub fn sum(&self, dims: Option<&[usize]>, keep_dim: bool) {
        
    }
}

impl std::ops::Add for &CPUTensor<f32> {
    type Output = CPUTensor<f32>;

    fn add(self, rhs: Self) -> Self::Output {
        let (a, b) = broadcast(self, rhs).expect(&format!(
            "Two tensors with shapes ({}, {}) cannot be broadcast.",
            self.shape(),
            rhs.shape()
        ));

        let dims = a.dims();
        let out = CPUTensor::from_shape(&a.shape());

        unsafe {
            let a_ptr = a.borrow().as_ptr() as *const libc::c_float;
            let b_ptr = b.borrow().as_ptr() as *const libc::c_float;
            let out_ptr = out.borrow_mut().as_mut_ptr() as *mut libc::c_float;
            interface::cpu_tensor_add_f32(
                a_ptr.add(a.offset()),
                b_ptr.add(b.offset()),
                out_ptr.add(out.offset()),
                a.stride().as_ptr(),
                b.stride().as_ptr(),
                out.stride().as_ptr(),
                a.shape().as_ptr(),
                dims,
                a.shape().numel(),
            );
        };

        out
    }
}

impl std::ops::AddAssign<&CPUTensor<f32>> for CPUTensor<f32> {
    fn add_assign(&mut self, rhs: &CPUTensor<f32>) {
        let b = rhs.broadcast_to(&self.shape()).expect(&format!(
            "RHS tensor with the shape {} cannot be broadcast to the shape {}.",
            rhs.shape(),
            self.shape()
        ));

        let dims = self.dims();
        unsafe {
            let self_ptr = self.borrow_mut().as_mut_ptr() as *mut libc::c_float;
            let b_ptr = b.borrow().as_ptr() as *const libc::c_float;
            interface::cpu_tensor_add_f32_(
                self_ptr.add(self.offset()),
                b_ptr.add(b.offset()),
                self.stride().as_ptr(),
                b.stride().as_ptr(),
                self.shape().as_ptr(),
                dims,
                self.shape().numel(),
            );
        };
    }
}

impl std::ops::Add for &CPUGenericTensor {
    type Output = CPUGenericTensor;

    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (CPUGenericTensor::F32(lhs), CPUGenericTensor::F32(rhs)) => {
                CPUGenericTensor::F32(lhs + rhs)
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

pub struct AddGrad {
    a_shape: Option<Shape>,
    b_shape: Option<Shape>
}

impl AddGrad {
    pub fn reduce_grad(grad: &CPUGenericTensor, target_shape: &Shape) -> CPUGenericTensor{
        todo!()
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
        let a_shape = self.a_shape.take().expect("AddGrad forward hasn't been called.");
        let b_shape = self.b_shape.take().expect("AddGrad forward hasn't been called.");

        return vec![AddGrad::reduce_grad(grad_inputs, &a_shape), AddGrad::reduce_grad(grad_inputs, &b_shape)];
    }
}

impl std::ops::Add for &CPUGraphNode {
    type Output = CPUGraphNode;

    fn add(self, rhs: Self) -> Self::Output {
        todo!()
    }
}
