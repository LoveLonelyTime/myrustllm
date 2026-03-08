use crate::common::{DType, Device, Shape, Tensor};

#[derive(Debug)]
pub struct GPUGenericTensor{

}

impl Tensor for GPUGenericTensor {
    fn shape(&self) -> Shape {
        todo!()
    }

    fn device(&self) -> Device {
        todo!()
    }

    fn dtype(&self) -> DType {
        todo!()
    }
}

impl Clone for GPUGenericTensor {
    fn clone(&self) -> Self {
        todo!()
    }
}