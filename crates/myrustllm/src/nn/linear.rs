use crate::{
    common::{DType, Device, GenericTensor, Shape, autograd::TensorGrad, math::TensorTranspose},
    nn::{module::Forward, parameter::Parameter},
};
use myrustllm_derive::Module;

use crate::common::math::TensorMatmul;

#[derive(Module)]
pub struct Linear {
    #[module]
    pub weight: Parameter,
    #[module]
    pub bias: Option<Parameter>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // TODO: randomize
        let weight = Parameter::new(TensorGrad::leaf(
            GenericTensor::zeros(
                &Shape::from([out_features, in_features]),
                DType::F32,
                Device::CPU,
            ),
            true,
        ));

        let bias = if bias {
            Some(Parameter::new(TensorGrad::leaf(
                GenericTensor::zeros(&Shape::from([out_features]), DType::F32, Device::CPU),
                true,
            )))
        } else {
            None
        };

        Linear { weight, bias }
    }
}

impl Forward<TensorGrad> for Linear {
    type Output = TensorGrad;
    fn forward(&self, input: TensorGrad) -> Self::Output {
        let output_mul = input.matmul(&self.weight.transpose());

        if let Some(bias) = &self.bias {
            &output_mul + &bias
        } else {
            output_mul
        }
    }
}
