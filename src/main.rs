use myrustllm::cpu::autograd::CPUGraphNode;
use myrustllm::cpu::dynamic::CPUGenericTensor;
use myrustllm::cpu::interface;
use myrustllm::cpu::math::TensorAddReduceBase;
use myrustllm::cpu::shape::Shape;
use myrustllm::cpu::slice::{Expand, TensorIndex};
use myrustllm::cpu::tensor::{CPUTensor, Tensor};
use myrustllm::cuda::tensor::CUDATensor;
use myrustllm::idx;
use std::rc::Rc;
use std::time::Instant;

fn main() {
    let n1 = CPUGraphNode::leaf(CPUTensor::<f32>::from_literal([1.0, 2.0, 3.0, 4.0]).into(), Some(Box::new(|t, g| {
        match g {
            CPUGenericTensor::F32(x) => { println!("{}", x)}
            _ => todo!()
        }
    })));
    let n2 = CPUGraphNode::leaf(CPUTensor::<f32>::from_literal([5.0]).into(), Some(Box::new(|t, g| {
        match g {
            CPUGenericTensor::F32(x) => { println!("{}", x)}
            _ => todo!()
        }
    })));
    let sum = &n1 + &n2;
    sum.backward();
}
