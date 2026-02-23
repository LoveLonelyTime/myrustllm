use myrustllm::cpu::shape::Shape;
use myrustllm::cpu::tensor::CPUTensor;
use myrustllm::cuda::tensor::CUDATensor;

fn main() {
    // let tensor1_cpu = CPUTensor::<f32>::from_array([
    //     [1.0, 2.0],
    //     [3.0, 4.0]
    // ]).slice(&idx!(.., 1));

    // let tensor2_cpu = CPUTensor::<f32>::from_array([
    //     [1.0, 2.0],
    //     [3.0, 4.0]
    // ]).slice(&idx!(.., 1));

    let tensor1_cpu = CPUTensor::<f32>::fill(&Shape::new(vec![1000,1000,100]), 1.0);
    let tensor2_cpu = CPUTensor::<f32>::fill(&Shape::new(vec![1000,1000,100]), 1.0);


    let tensor1_cuda = CUDATensor::from(&tensor1_cpu);
    let tensor2_cuda = CUDATensor::from(&tensor2_cpu);

    let res_cuda = &tensor1_cuda + &tensor2_cuda;

    let res = CPUTensor::from(&res_cuda);

    println!("{}", res);
}
