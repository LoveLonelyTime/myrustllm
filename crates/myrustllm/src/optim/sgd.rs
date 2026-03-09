use crate::{nn::{
    module::{Module, ParamCollector},
    parameter::Parameter,
}, optim::optimizer::Optimizer};

pub struct SGD<'a> {
    pub params: Vec<&'a mut Parameter>,
    pub lr: f32,
    pub momentum: Option<f32>,
}

impl<'a> SGD<'a> {
    pub fn new(module: &'a mut dyn Module, lr: f32, momentum: Option<f32>) -> Self {
        let mut para_collector = ParamCollector::new();
        module.visit(&mut para_collector);

        SGD {
            params: para_collector.into(),
            lr,
            momentum,
        }
    }
}

impl <'a> Optimizer for SGD<'a> {
    fn step(&mut self) {
        todo!()
    }

    fn zero_grad(&mut self) {
        todo!()
    }
}
