use std::ops::{Deref, DerefMut};

use crate::common::autograd::TensorGrad;
use crate::nn::module::{Module, ModuleVisitor};

pub struct Parameter {
    tensor: TensorGrad,
}

impl Parameter {
    pub fn new(tensor: TensorGrad) -> Self {
        Parameter { tensor }
    }
}

impl Module for Parameter {
    fn apply<'a>(&'a mut self, visitor: &mut dyn ModuleVisitor<'a>) {
        visitor.visit_parameter(self);
    }
}

impl Deref for Parameter {
    type Target = TensorGrad;

    fn deref(&self) -> &Self::Target {
        &self.tensor
    }
}

impl DerefMut for Parameter {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.tensor
    }
}
