use crate::common::dtype::{Any, DTYPE_F32};
use crate::common::init::{TensorAllocInit, TensorOnesInit, TensorRawDataInit, TensorZerosInit};
use crate::common::io::TensorRawData;
use crate::common::ops::cast::TensorCast;
use crate::common::shape::create_contiguous_stride;
use crate::common::{DTypeImpl, Impl, Shape};
use crate::cpu::impls::{CPU, CPUTensorAnyPrototype, CPUTensorPrototype};
use crate::cpu::interface::IntoInterface;
use crate::cpu::mem::CPUMemory;
use num_traits::{One, Zero};

impl<T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<U>>, U> TensorAllocInit<CPU> for T {
    fn init(shape: &Shape, _device: &<CPU as Impl>::Device) -> Self::Prototype {
        CPUTensorPrototype::alloc(shape)
    }
}

impl<T> CPUTensorPrototype<T> {
    pub fn fill(value: T, shape: &Shape) -> Self {
        todo!()
    }
}

impl<T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<U>>, U: Zero> TensorZerosInit<CPU> for T {
    fn init(shape: &Shape, _device: &<CPU as Impl>::Device) -> Self::Prototype {
        CPUTensorPrototype::fill(U::zero(), shape)
    }
}

impl<T: DTypeImpl<CPU, Prototype = CPUTensorPrototype<U>>, U: One> TensorOnesInit<CPU> for T {
    fn init(shape: &Shape, _device: &<CPU as Impl>::Device) -> Self::Prototype {
        CPUTensorPrototype::fill(U::one(), shape)
    }
}

impl TensorRawDataInit<CPU> for Any {
    fn init(data: impl Into<TensorRawData>, _device: &<CPU as Impl>::Device) -> Self::Prototype {
        let mut data = data.into();
        let shape = data.shape;

        // TODO: Big-endian
        match data.dtype {
            DTYPE_F32 => {
                let mut mem: CPUMemory<f32> = CPUMemory::new(shape.numel());
                if let Err(e) = data.source.read(unsafe {
                    std::slice::from_raw_parts_mut(
                        mem.as_mut_ptr() as *mut u8,
                        mem.size() * size_of::<f32>(),
                    )
                }) {
                    panic!("Cannot create a tensor from raw data: {}.", e);
                }

                CPUTensorAnyPrototype::from(CPUTensorPrototype::new(
                    mem.into(),
                    &shape,
                    &create_contiguous_stride(&shape),
                    0,
                ))
            }
            _ => panic!("DType {} is not supported.", data.dtype),
        }
    }
}

impl<T: DTypeImpl<CPU, Prototype: IntoInterface> + TensorAllocInit<CPU>> TensorRawDataInit<CPU>
    for T
{
    fn init(data: impl Into<TensorRawData>, device: &<CPU as Impl>::Device) -> Self::Prototype {
        <Any as TensorCast<CPU, T>>::cast(&<Any as TensorRawDataInit<CPU>>::init(data, device))
    }
}
