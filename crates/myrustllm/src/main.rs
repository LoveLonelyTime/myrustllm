use myrustllm::common::autograd::TensorGrad;
use myrustllm::common::{DType, Device, GenericTensor, Shape};
use myrustllm::cpu::dynamic::CPUGenericTensor;
use myrustllm::cpu::interface;
use myrustllm::cpu::math::TensorAddReduceBase;
use myrustllm::cpu::slice::{Expand, TensorIndex};
use myrustllm::cpu::tensor::CPUTensor;
use myrustllm::cuda::tensor::CUDATensor;
use myrustllm::nn::linear::Linear;
use myrustllm::nn::module::Forward;
use myrustllm::{cascade, idx, no_grad};
use std::rc::Rc;
use std::time::Instant;

fn main() {
    let n1 = TensorGrad::leaf(
        myrustllm::common::GenericTensor::CPUTensor(CPUTensor::<f32>::from_literal([1.0]).into()),
        true,
    );

    // let n2 = TensorGrad::leaf(
    //     myrustllm::common::GenericTensor::CPUTensor(CPUGenericTensor::F32(
    //         CPUTensor::<f32>::from_literal([1.0]),
    //     )),
    //     true,
    // );

    let init_grad = TensorGrad::leaf(
        GenericTensor::fill_ones(&Shape::from([512, 512]), DType::F32, Device::CPU),
        true,
    );

    // let mut k = &n1 + &n2;

    let l1 = Linear::new(512, 512, true);
    let l2 = Linear::new(512, 512, true);
    let l3 = Linear::new(512, 512, true);
    let l4 = Linear::new(512, 512, true);
    let m = cascade!(l1, l2, l3, l4);

    let mut output = m.forward(n1.clone());

    no_grad!({
        output.backward(&init_grad, true);
    });

    println!("{:?}", n1.grad().as_ref().unwrap().tensor())
}
