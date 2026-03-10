use crate::common::dtype::Scalar;
use crate::common::{DType, Device, GenericTensor, Shape, Tensor};
use crate::cpu::dynamic::CPUGenericTensor;

impl GenericTensor {
    // ================================================== ALLOC ==================================================

    pub fn alloc(shape: &Shape, dtype: DType, device: Device) -> Self {
        match device {
            Device::CPU => CPUGenericTensor::alloc(shape, dtype).into(),
            _ => todo!(),
        }
    }

    pub fn alloc_like(tensor: &GenericTensor) -> Self {
        GenericTensor::alloc(&tensor.shape(), tensor.dtype(), tensor.device())
    }

    // ================================================== FILL ==================================================

    pub fn fill(shape: &Shape, val: Scalar, device: Device) -> Self {
        match device {
            Device::CPU => CPUGenericTensor::fill(shape, val).into(),
            _ => todo!(),
        }
    }

    pub fn fill_(&mut self, val: Scalar) {
        match self {
            GenericTensor::CPUTensor(t) => t.fill_(val),
            _ => todo!(),
        }
    }

    pub fn fill_like(tensor: &GenericTensor, val: Scalar) -> Self {
        GenericTensor::fill(&tensor.shape(), val, tensor.device())
    }

    pub fn scalar(val: Scalar, device: Device) -> Self {
        GenericTensor::fill(&Shape::scalar(), val, device)
    }

    pub fn zeros(shape: &Shape, dtype: DType, device: Device) -> Self {
        match device {
            Device::CPU => CPUGenericTensor::zeros(shape, dtype).into(),
            _ => todo!(),
        }
    }

    pub fn zeros_(&mut self) {
        match self {
            GenericTensor::CPUTensor(t) => t.zeros_(),
            _ => todo!(),
        }
    }

    pub fn zeros_like(tensor: &GenericTensor) -> Self {
        GenericTensor::zeros(&tensor.shape(), tensor.dtype(), tensor.device())
    }

    pub fn ones(shape: &Shape, dtype: DType, device: Device) -> Self {
        match device {
            Device::CPU => CPUGenericTensor::ones(shape, dtype).into(),
            _ => todo!(),
        }
    }

    pub fn ones_(&mut self) {
        match self {
            GenericTensor::CPUTensor(t) => t.ones_(),
            _ => todo!(),
        }
    }

    pub fn ones_like(tensor: &GenericTensor) -> Self {
        GenericTensor::ones(&tensor.shape(), tensor.dtype(), tensor.device())
    }

    // ================================================== NORMAL ==================================================
    pub fn uniform(shape: &Shape, mean: f32, std: f32, dtype: DType, device: Device) -> Self {
        match device {
            Device::CPU => CPUGenericTensor::uniform(shape, mean, std, dtype).into(),
            _ => todo!(),
        }
    }

    pub fn uniform_(&mut self, mean: f32, std: f32) {
        match self {
            GenericTensor::CPUTensor(t) => t.uniform_(mean, std),
            _ => todo!(),
        }
    }

    pub fn uniform_like(tensor: &GenericTensor, mean: f32, std: f32) -> Self {
        GenericTensor::uniform(&tensor.shape(), mean, std, tensor.dtype(), tensor.device())
    }
}
