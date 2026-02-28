use crate::common::{DType, Device, Shape, Tensor};
use crate::cpu::dynamic::CPUGenericTensor;
use crate::cuda::dynamic::GPUGenericTensor;

/// Generic tensor.
/// 
/// `GenericTensor` represents all tensors (dispatching device, dtype).
pub enum GenericTensor {
    CPUTensor(CPUGenericTensor),
    GPUTensor(GPUGenericTensor)
}

impl Tensor for GenericTensor {
    fn shape(&self) -> Shape {
        match self {
            GenericTensor::CPUTensor(t) => t.shape(),
            GenericTensor::GPUTensor(t) => t.shape()
        }
    }

    fn device(&self) -> Device {
        match self {
            GenericTensor::CPUTensor(t) => t.device(),
            GenericTensor::GPUTensor(t) => t.device()
        }
    }

    fn dtype(&self) -> DType {
        match self {
            GenericTensor::CPUTensor(t) => t.dtype(),
            GenericTensor::GPUTensor(t) => t.dtype()
        }
    }
}

impl Clone for GenericTensor {
    fn clone(&self) -> Self {
        match self {
            GenericTensor::CPUTensor(t) => GenericTensor::CPUTensor(t.clone()),
            GenericTensor::GPUTensor(t) => GenericTensor::GPUTensor(t.clone())
        }
    }
}

impl GenericTensor {
    pub fn like_ones(tensor: &GenericTensor) -> Self {
        match tensor {
            _ => todo!()
        }
    }
}