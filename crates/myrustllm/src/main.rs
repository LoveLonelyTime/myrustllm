use myrustllm::common::autograd::TensorGrad;
use myrustllm::common::math::{TensorAddReduce, TensorMatmul};
use myrustllm::common::{DType, Device, GenericTensor, Shape};
use myrustllm::cpu::dynamic::CPUGenericTensor;
use myrustllm::cpu::init::TensorFillInitBase;
use myrustllm::cpu::interface;
use myrustllm::cpu::math::{TensorAddReduceBase, TensorMatmulBase};
use myrustllm::cpu::slice::{Expand, TensorIndex};
use myrustllm::cpu::tensor::CPUTensor;
use myrustllm::cuda::tensor::CUDATensor;
use myrustllm::nn::init::Initializer;
use myrustllm::nn::linear::Linear;
use myrustllm::nn::module::{Forward, Module, ParamCollector};
use myrustllm::nn::optimizer::{Optimizer, SGD};
use myrustllm::{cascade, idx, no_grad};
use std::rc::Rc;
use std::time::Instant;

fn main() {

    // let n1 = TensorGrad::leaf(
    //     myrustllm::common::GenericTensor::CPUTensor(CPUTensor::<f32>::from_literal([1.0]).into()),
    //     true,
    // );

    // let n2 = TensorGrad::leaf(
    //     myrustllm::common::GenericTensor::CPUTensor(CPUGenericTensor::F32(
    //         CPUTensor::<f32>::from_literal([1.0]),
    //     )),
    //     true,
    // );

    let input = TensorGrad::leaf(
        GenericTensor::ones(&Shape::from([64, 512]), DType::F32, Device::CPU),
        true,
    );

    let init_grad = TensorGrad::leaf(
        GenericTensor::ones(&Shape::from([64, 512]), DType::F32, Device::CPU),
        true,
    );

    // let mut k = &n1 + &n2;

    let l1 = Linear::new(512, 512, true);
    let l2 = Linear::new(512, 512, true);
    let l3 = Linear::new(512, 512, true);
    let l4 = Linear::new(512, 512, true);
    let mut m = cascade!(l1, l2, l3, l4);

    let mut init = Initializer::normal(1.0, 0.0);
    m.apply(&mut init);

    let mut output = m.forward(input.clone());
    // let mut output = output.add_reduce(None, false);
    // println!("{:?}", output.tensor());

    let mut optim = SGD::new(&mut m, 0.1, Some(123.0));
    optim.zero_grad();

    no_grad!({
        output.backward(&init_grad, true);
    });

    optim.step();

    // let mut output = m.forward(n1.clone());

    // no_grad!({
    //     output.backward(&init_grad, true);
    // });

    // println!("{:?}", input.grad())

    // println!("{:?}", n1.grad().as_ref().unwrap().tensor())

    // let start = Instant::now();
    // let lhs = CPUTensor::<f32>::fill_ones(&Shape::from([64, 1024, 1024]));
    // let duration = start.elapsed();
    // println!("执行时间: {:?}", duration);

    // let rhs = CPUTensor::<f32>::fill_ones(&Shape::from([64, 1024, 1024]));
    // // let lhs = CPUTensor::from_literal([[[1.0, 2.0], [3.0, 4.0]]]);
    // // let rhs = CPUTensor::from_literal([[[1.0, 2.0], [3.0, 4.0]]]);
    // let start = Instant::now();
    // let out = lhs.matmul(&rhs);
    // let duration = start.elapsed();
    // println!("执行时间: {:?}", duration);
    // println!("{:?}", out)
}
