use std::rc::Rc;

use num_traits::{One, Zero};

use crate::{
    common::{
        DType, Shape,
        dtype::{Any, DTypeImpl, F32, F64, I32, StdDType},
        impls::Impl,
        init::{TensorAllocInit, TensorOnesInit, TensorRawDataInit, TensorZerosInit},
        io::TensorRawData,
        ops::cast::TensorCastImpl,
    },
    cpu::{
        impls::{CPU, CPUTensorAnyPrototype, CPUTensorPrototype},
        mem::CPUMemory,
    },
};

impl<T: StdDType> TensorAllocInit<CPU> for T {
    fn init(shape: &Shape, _device: &<CPU as Impl>::Device) -> Self::Prototype {
        CPUTensorPrototype::alloc(shape)
    }
}

impl<T: Copy> CPUTensorPrototype<T> {
    pub fn fill(value: T, shape: &Shape) -> Self {
        todo!()
    }
}

impl<T: StdDType<RType: Zero>> TensorZerosInit<CPU> for T {
    fn init(shape: &Shape, _device: &<CPU as Impl>::Device) -> Self::Prototype {
        CPUTensorPrototype::fill(T::RType::zero(), shape)
    }
}

impl<T: StdDType<RType: One>> TensorOnesInit<CPU> for T {
    fn init(shape: &Shape, _device: &<CPU as Impl>::Device) -> Self::Prototype {
        CPUTensorPrototype::fill(T::RType::one(), shape)
    }
}

impl TensorRawDataInit<CPU> for Any {
    fn init(data: impl Into<TensorRawData>, _device: &<CPU as Impl>::Device) -> Self::Prototype {
        let data = data.into();
        let shape = data.shape;
        match data.dtype {
            DType::F32 => {
                let mut mem: CPUMemory<f32> = CPUMemory::new(shape.numel());
                data.source.read(0, unsafe {
                    std::slice::from_raw_parts_mut(mem.as_mut_ptr() as *mut u8, data.source.len())
                });

                CPUTensorAnyPrototype::from(CPUTensorPrototype::new(
                    mem.into(),
                    &shape,
                    &Shape::create_contiguous_stride(&shape),
                    0,
                ))
            }
            DType::F64 => {
                let mut mem: CPUMemory<f64> = CPUMemory::new(shape.numel());
                data.source.read(0, unsafe {
                    std::slice::from_raw_parts_mut(mem.as_mut_ptr() as *mut u8, data.source.len())
                });

                CPUTensorAnyPrototype::from(CPUTensorPrototype::new(
                    mem.into(),
                    &shape,
                    &Shape::create_contiguous_stride(&shape),
                    0,
                ))
            }
            DType::I32 => {
                let mut mem: CPUMemory<i32> = CPUMemory::new(shape.numel());
                data.source.read(0, unsafe {
                    std::slice::from_raw_parts_mut(mem.as_mut_ptr() as *mut u8, data.source.len())
                });

                CPUTensorAnyPrototype::from(CPUTensorPrototype::new(
                    mem.into(),
                    &shape,
                    &Shape::create_contiguous_stride(&shape),
                    0,
                ))
            }
            DType::I64 => {
                let mut mem: CPUMemory<i64> = CPUMemory::new(shape.numel());
                data.source.read(0, unsafe {
                    std::slice::from_raw_parts_mut(mem.as_mut_ptr() as *mut u8, data.source.len())
                });

                CPUTensorAnyPrototype::from(CPUTensorPrototype::new(
                    mem.into(),
                    &shape,
                    &Shape::create_contiguous_stride(&shape),
                    0,
                ))
            }
            _ => panic!(""),
        }
    }
}

impl<T: StdDType> TensorRawDataInit<CPU> for T {
    fn init(data: impl Into<TensorRawData>, device: &<CPU as Impl>::Device) -> Self::Prototype {
        <Any as TensorCastImpl<CPU, T>>::cast(&<Any as TensorRawDataInit<CPU>>::init(data, device))
    }
}
