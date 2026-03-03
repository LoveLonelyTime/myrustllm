use crate::{common::{Shape, device::{self, Device}, dtype::DType, tensor::Tensor}, cpu::tensor::CPUTensor};

/// CPUGenericTensor
/// Compared to `CPUTensor<T>`, the data type of `CPUGenericTensor` is dispatched dynamically.
pub enum CPUGenericTensor {
    F32(CPUTensor<f32>),
    F64(CPUTensor<f64>),
    I32(CPUTensor<i32>),
    I64(CPUTensor<i64>),
}

impl Tensor for CPUGenericTensor {
    fn shape(&self) -> Shape {
        match self {
            CPUGenericTensor::F32(t) => t.shape(),
            CPUGenericTensor::F64(t) => t.shape(),
            CPUGenericTensor::I32(t) => t.shape(),
            CPUGenericTensor::I64(t) => t.shape(),
        }
    }

    fn device(&self) -> Device {
        match self {
            CPUGenericTensor::F32(t) => t.device(),
            CPUGenericTensor::F64(t) => t.device(),
            CPUGenericTensor::I32(t) => t.device(),
            CPUGenericTensor::I64(t) => t.device(),
        }
    }

    fn dtype(&self) -> DType {
        match self {
            CPUGenericTensor::F32(t) => t.dtype(),
            CPUGenericTensor::F64(t) => t.dtype(),
            CPUGenericTensor::I32(t) => t.dtype(),
            CPUGenericTensor::I64(t) => t.dtype(),
        }
    }
}

impl Clone for CPUGenericTensor {
    fn clone(&self) -> Self {
        match self {
            CPUGenericTensor::F32(t) => CPUGenericTensor::F32(t.clone()),
            CPUGenericTensor::F64(t) => CPUGenericTensor::F64(t.clone()),
            CPUGenericTensor::I32(t) => CPUGenericTensor::I32(t.clone()),
            CPUGenericTensor::I64(t) => CPUGenericTensor::I64(t.clone()),
        }
    }
}

impl From<CPUTensor<f32>> for CPUGenericTensor {
    fn from(value: CPUTensor<f32>) -> Self {
        CPUGenericTensor::F32(value)
    }
}

impl From<CPUTensor<f64>> for CPUGenericTensor {
    fn from(value: CPUTensor<f64>) -> Self {
        CPUGenericTensor::F64(value)
    }
}

impl CPUGenericTensor {
    pub fn like_zeros(tensor: &CPUGenericTensor) -> Self {
        match tensor {
            CPUGenericTensor::F32(t) => {
                CPUGenericTensor::F32(CPUTensor::<f32>::fill_zeros(&t.shape()))
            }
            _ => {
                todo!()
            }
        }
    }

    pub fn like_ones(tensor: &CPUGenericTensor) -> Self {
        match tensor {
            CPUGenericTensor::F32(t) => {
                CPUGenericTensor::F32(CPUTensor::<f32>::fill_ones(&t.shape()))
            }
            _ => {
                todo!()
            }
        }
    }
}
