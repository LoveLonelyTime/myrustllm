use crate::common::{Shape, DType, Device, Tensor};
use crate::cpu::tensor::CPUTensor;

/// CPUGenericTensor
/// Compared to `CPUTensor<T>`, the data type of `CPUGenericTensor` is dispatched dynamically.
#[derive(Debug, Clone)]
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

impl From<CPUTensor<i32>> for CPUGenericTensor {
    fn from(value: CPUTensor<i32>) -> Self {
        CPUGenericTensor::I32(value)
    }
}

impl From<CPUTensor<i64>> for CPUGenericTensor {
    fn from(value: CPUTensor<i64>) -> Self {
        CPUGenericTensor::I64(value)
    }
}

// Create
impl CPUGenericTensor {
    pub fn fill_zeros(shape: &Shape, dtype: DType) -> Self {
        match dtype {
            DType::F32 => CPUTensor::<f32>::fill_zeros(shape).into(),
            _ => todo!(),
        }
    }

    pub fn fill_ones(shape: &Shape, dtype: DType) -> Self {
        match dtype {
            DType::F32 => CPUTensor::<f32>::fill_ones(shape).into(),
            _ => todo!(),
        }
    }

    pub fn like_zeros(tensor: &CPUGenericTensor) -> Self {
        CPUGenericTensor::fill_zeros(&tensor.shape(), tensor.dtype())
    }

    pub fn like_ones(tensor: &CPUGenericTensor) -> Self {
        CPUGenericTensor::fill_ones(&tensor.shape(), tensor.dtype())
    }
}
