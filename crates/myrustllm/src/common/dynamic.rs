use crate::common::{DType, Device, Shape, Tensor};
use crate::cpu::dynamic::CPUGenericTensor;
use crate::cuda::dynamic::GPUGenericTensor;

/// Generic tensor.
///
/// `GenericTensor` represents all tensors (dispatching device, dtype).
#[derive(Clone, Debug)]
pub enum GenericTensor {
    CPUTensor(CPUGenericTensor),
    GPUTensor(GPUGenericTensor),
}

impl From<CPUGenericTensor> for GenericTensor {
    fn from(value: CPUGenericTensor) -> Self {
        GenericTensor::CPUTensor(value)
    }
}

impl From<GPUGenericTensor> for GenericTensor {
    fn from(value: GPUGenericTensor) -> Self {
        GenericTensor::GPUTensor(value)
    }
}

impl Tensor for GenericTensor {
    fn shape(&self) -> Shape {
        match self {
            GenericTensor::CPUTensor(t) => t.shape(),
            GenericTensor::GPUTensor(t) => t.shape(),
        }
    }

    fn device(&self) -> Device {
        match self {
            GenericTensor::CPUTensor(t) => t.device(),
            GenericTensor::GPUTensor(t) => t.device(),
        }
    }

    fn dtype(&self) -> DType {
        match self {
            GenericTensor::CPUTensor(t) => t.dtype(),
            GenericTensor::GPUTensor(t) => t.dtype(),
        }
    }
}