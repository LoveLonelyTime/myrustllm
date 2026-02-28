use crate::common::GenericTensor;

pub trait TensorCopy<Rhs = Self> {
    fn copy_from(&mut self, rhs: Rhs);
}

impl TensorCopy for &GenericTensor {
    fn copy_from(&mut self, rhs: Self) {
        match (self, rhs) {
            _ => todo!()
        }
    }
}

pub trait TensorAddReduce{
    type Output;
    fn add_reduce(self, dims: Option<&[usize]>, keep_dim: bool) -> Self::Output;
}
