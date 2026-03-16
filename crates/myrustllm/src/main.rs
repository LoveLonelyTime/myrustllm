use myrustllm::autograd::impls::Autograd;
use myrustllm::common::Shape;
use myrustllm::common::dtype::{Any, F32, F64, I32};
use myrustllm::cpu::impls::CPU;

fn main() {
    let s: [i32; 0] = [];
    println!("{:?}", s.iter().product::<i32>());
}
