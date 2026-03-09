pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}