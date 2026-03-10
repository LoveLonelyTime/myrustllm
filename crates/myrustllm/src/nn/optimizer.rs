use crate::{
    common::{GenericTensor, dtype::Scalar},
    nn::{
        module::{Module, ParamCollector},
        parameter::Parameter,
    },
};

use crate::common::Tensor;

pub trait Optimizer {
    fn step(&mut self);
    fn zero_grad(&mut self);
}

pub struct SGD<'a> {
    params: Vec<&'a mut Parameter>,
    lr: f32,
    momentum: Option<f32>,
    states: Vec<Option<GenericTensor>>,
}

impl<'a> SGD<'a> {
    pub fn new(module: &'a mut impl Module, lr: f32, momentum: Option<f32>) -> Self {
        let mut param_collector = ParamCollector::new();
        module.apply(&mut param_collector);
        let params: Vec<&mut Parameter> = param_collector.into();
        let num_param = params.len();

        SGD {
            params,
            lr,
            momentum,
            states: vec![None; num_param],
        }
    }
}

impl Optimizer for SGD<'_> {
    fn step(&mut self) {
        for (param, state) in self.params.iter_mut().zip(self.states.iter_mut()) {
            let grad = param.grad().unwrap().tensor().clone();
            let mut update = grad.clone();

            if let Some(mu) = self.momentum {
                let buf = state.get_or_insert(GenericTensor::zeros_like(&grad));
                *buf = &(&*buf * &GenericTensor::scalar(Scalar::F32(mu), buf.device())) + &update;
                update = buf.clone();
            }
            let mut param = param.tensor().clone();
            param -= &(&update * &GenericTensor::scalar(Scalar::F32(self.lr), update.device()));
        }
    }

    fn zero_grad(&mut self) {
        for param in &mut self.params {
            param.zero_grad();
        }
    }
}
