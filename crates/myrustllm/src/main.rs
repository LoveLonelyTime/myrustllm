use myrustllm::{
    common::{
        Shape, Tensor,
        dtype::{F32, F64, I32},
        ops::cast::TensorCast,
    },
    cpu::impls::CPU,
};

fn main() {
    let t = Tensor::<CPU, F32>::ones(&Shape::from([500]), &());
    let t2 = Tensor::<CPU, F32>::from_literal([32f32], &());

    println!("t: {}", &t / &t2);
}
