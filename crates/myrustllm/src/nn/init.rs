use crate::common::GenericTensor;
use crate::nn::module::{Module, ModuleVisitor};
use crate::nn::parameter::Parameter;

pub struct Initializer {
    init_func: Box<dyn FnMut(&mut GenericTensor) -> ()>,
}

impl Initializer {
    pub fn new(init_func: impl FnMut(&mut GenericTensor) -> () + 'static) -> Self {
        Initializer {
            init_func: Box::new(init_func),
        }
    }

    pub fn zeros() -> Self {
        Initializer::new(move |t| t.zeros_())
    }

    pub fn ones() -> Self {
        Initializer::new(move |t| t.ones_())
    }

    pub fn normal(mean: f32, std: f32) -> Self {
        Initializer::new(move |t| t.uniform_(mean, std))
    }
}

impl ModuleVisitor<'_> for Initializer {
    fn visit_parameter(&mut self, param: &'_ mut Parameter) {
        (self.init_func)(&mut param.tensor().clone());
    }
}
