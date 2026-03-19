use myrustllm::{
    autograd::impls::Autograd,
    common::{Shape, Tensor, dtype::F32},
    cpu::impls::CPU,
};

fn main() {
    // let t = Tensor::<Autograd<CPU, F32>, F32>::fill(1.0, &Shape::from([500, 500]), &());
    let mut t2 = Tensor::<Autograd<CPU, F32>, F32>::ones(&Shape::from([500, 500]), &());
    t2.prototype.catch_grad();
    let t = Tensor::<Autograd<CPU, F32>, F32>::ones(&Shape::from([500, 500]), &());
    let init = Tensor::<Autograd<CPU, F32>, F32>::ones(&Shape::from([500, 500]), &());
    (&t + &t2).backward(&init, false);
    let k = t2.prototype.grad();
    let p = k.grad().unwrap();
    println!("{}", p.prototype.tensor())
}
