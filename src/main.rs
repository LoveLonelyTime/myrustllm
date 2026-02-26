use myrustllm::cpu::shape::Shape;
use myrustllm::cpu::tensor::{CPUTensor, Tensor};
use myrustllm::cuda::tensor::CUDATensor;
use myrustllm::{idx};
use myrustllm::cpu::slice::{Expand, TensorIndex};
use std::time::Instant;

fn main() {
    // let tensor1_cpu = CPUTensor::<f32>::from_array([
    //     [1.0, 2.0],
    //     [3.0, 4.0]
    // ]).slice(&idx!(.., 1));

    // let tensor2_cpu = CPUTensor::<f32>::from_array([
    //     [1.0, 2.0],
    //     [3.0, 4.0]
    // ]).slice(&idx!(.., 1));

    //let tensor1_cpu = CPUTensor::<f32>::fill(&Shape::new(vec![1000,1000,100]), 1.0);
    //let tensor2_cpu = CPUTensor::<f32>::fill(&Shape::new(vec![1]), 2.0);


    // let mut  tensor1_cuda = CUDATensor::from(&tensor1_cpu);
    // let tensor2_cuda = CUDATensor::from(&tensor2_cpu);

    // let res_cuda = -&tensor1_cuda;
    // tensor1_cuda.slice(&idx!(1,3..50,30..50)).copy_from(&tensor2_cuda);

    let tensor = CPUTensor::<f32>::from_shape(&Shape::new(vec![1000, 1000, 1000]));
    // let new_tensor = tensor.broadcast_to(&Shape::new(vec![1,2,3])).unwrap();
    let start = Instant::now();
   //  let res = tensor.sum(Some(&vec![0,2]), true);
    let duration = start.elapsed();
    println!("执行时间: {:?}", duration);
   // println!("{}", res.shape());
}
