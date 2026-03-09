use std::ops::Deref;

use crate::common::autograd::TensorGrad;
use crate::nn::module::{Module, ModuleVisitor};

pub struct Parameter {
    pub tensor: TensorGrad,
}

impl Parameter {
    pub fn new(tensor: TensorGrad) -> Self{
        Parameter { tensor }
    }
}

impl Module for Parameter {
    fn visit<'a>(&'a mut self, visitor: &mut dyn ModuleVisitor<'a>) {
        visitor.visit_parameter(self);
    }
}

impl Deref for Parameter {
    type Target = TensorGrad;
    
    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}
