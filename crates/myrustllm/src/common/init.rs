use crate::common::{Shape, Tensor};
use crate::common::{dtype::DTypeImpl, impls::Impl};

use crate::common::io::TensorRawData;

pub trait TensorAllocInit<I: Impl>: DTypeImpl<I> {
    fn init(shape: &Shape, device: &I::Device) -> Self::Prototype;
}

impl<I: Impl, TI: DTypeImpl<I> + TensorAllocInit<I>> Tensor<I, TI> {
    pub fn alloc(shape: &Shape, device: &I::Device) -> Self {
        Tensor::new(TI::init(shape, device))
    }
}

pub trait TensorZerosInit<I: Impl>: DTypeImpl<I> {
    fn init(shape: &Shape, device: &I::Device) -> Self::Prototype;
}

impl<I: Impl, TI: DTypeImpl<I> + TensorZerosInit<I>> Tensor<I, TI> {
    pub fn zeros(shape: &Shape, device: &I::Device) -> Self {
        Tensor::new(TI::init(shape, device))
    }
}

pub trait TensorOnesInit<I: Impl>: DTypeImpl<I> {
    fn init(shape: &Shape, device: &I::Device) -> Self::Prototype;
}

impl<I: Impl, TI: DTypeImpl<I> + TensorOnesInit<I>> Tensor<I, TI> {
    pub fn ones(shape: &Shape, device: &I::Device) -> Self {
        Tensor::new(TI::init(shape, device))
    }
}

pub trait TensorRawDataInit<I: Impl>: DTypeImpl<I> {
    fn init(data: impl Into<TensorRawData>, device: &I::Device) -> Self::Prototype;
}

impl<I: Impl, TI: DTypeImpl<I> + TensorRawDataInit<I>> Tensor<I, TI> {
    pub fn from_raw_data(data: impl Into<TensorRawData>, device: &I::Device) -> Self {
        Tensor::new(TI::init(data, device))
    }
}
