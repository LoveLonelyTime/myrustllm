use crate::{
    common::{DType, Device, GenericTensor, Shape, autograd::TensorGrad},
    nn::{module::Forward, parameter::Parameter},
};
use myrustllm_derive::Module;

#[derive(Module)]
pub struct Linear {
    #[module]
    weight: Parameter,
    #[module]
    bias: Option<Parameter>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        // TODO: randomize
        let weight = Parameter::new(TensorGrad::leaf(
            GenericTensor::fill_zeros(
                &Shape::from([out_features, in_features]),
                DType::F32,
                Device::CPU,
            ),
            true,
        ));

        let bias = if bias {
            Some(Parameter::new(TensorGrad::leaf(
                GenericTensor::fill_zeros(&Shape::from([out_features]), DType::F32, Device::CPU),
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
        // TODO: matmul
        let output_mul = &self.weight.tensor + &input;

        if let Some(bias) = &self.bias {
            &output_mul + &bias.tensor
        } else {
            output_mul
        }
    }
}
