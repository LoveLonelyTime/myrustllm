use std::{cell::RefCell, os::raw::c_void, rc::Rc};

use crate::{
    common::{
        DType, Shape,
        dtype::{Any, DTypeImpl, F32, F64, I32, I64, StdDType},
        impls::Impl,
        init::{TensorAllocInit, TensorRawDataInit},
        io::TensorRawData,
        tensor::TensorPrototype,
    },
    cpu::{
        interface::{self, IntoInterface},
        mem::{CPUMemory, SharedCPUMemory},
    },
};

pub struct CPU {}

impl Impl for CPU {
    type Device = ();
}

#[derive(Clone)]
pub struct CPUTensorPrototype<T> {
    data: SharedCPUMemory<T>,
    shape: Shape,
    stride: Shape,
    offset: usize,
}

impl<T> CPUTensorPrototype<T> {
    pub fn new(data: SharedCPUMemory<T>, shape: &Shape, stride: &Shape, offset: usize) -> Self {
        CPUTensorPrototype {
            data,
            shape: shape.clone(),
            stride: stride.clone(),
            offset,
        }
    }

    pub fn alloc(shape: &Shape) -> Self {
        CPUTensorPrototype {
            data: Rc::new(RefCell::new(CPUMemory::new(shape.numel()))),
            shape: shape.clone(),
            stride: Shape::create_contiguous_stride(shape),
            offset: 0,
        }
    }

    pub fn stride(&self) -> Shape {
        self.stride.clone()
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    pub fn data(&self) -> SharedCPUMemory<T> {
        self.data.clone()
    }
}

// For StdRType
impl<T: StdDType> TensorPrototype for CPUTensorPrototype<T::RType> {
    fn shape(&self) -> Shape {
        self.shape.clone()
    }

    fn dtype(&self) -> DType {
        T::DTYPE
    }
}

impl<T: StdRType> IntoInterface for CPUTensorPrototype<T> {
    /// Wrap `CPUTensor<T>` into an interface struct.
    unsafe fn into_interface(&self) -> interface::CPUTensor {
        interface::CPUTensor {
            data: unsafe { self.data.borrow_mut().as_mut_ptr().add(self.offset) as *mut u8 },
            shape: self.shape.as_ptr(),
            stride: self.stride.as_ptr(),
            dims: self.dims(),
            dtype: T::DTYPE as interface::DType,
        }
    }
}

// For stdDType
impl<T: StdDType> DTypeImpl<CPU> for T {
    type Prototype = CPUTensorPrototype<T::RType>;
}

// Earse-Type
#[derive(Clone)]
pub struct CPUTensorAnyPrototype {
    inner: CPUTensorPrototype<u8>,
    dtype: DType,
}

impl<T: StdRType> From<CPUTensorPrototype<T>> for CPUTensorAnyPrototype {
    fn from(value: CPUTensorPrototype<T>) -> Self {
        CPUTensorAnyPrototype {
            inner: unsafe { std::mem::transmute(value) },
            dtype: T::DTYPE,
        }
    }
}

impl TensorPrototype for CPUTensorAnyPrototype {
    fn shape(&self) -> Shape {
        self.inner.shape.clone()
    }

    fn dtype(&self) -> DType {
        self.dtype
    }
}

impl DTypeImpl<CPU> for Any {
    type Prototype = CPUTensorAnyPrototype;
}

impl IntoInterface for CPUTensorAnyPrototype {
    /// Wrap `CPUTensor<T>` into an interface struct.
    unsafe fn into_interface(&self) -> interface::CPUTensor {
        interface::CPUTensor {
            data: unsafe {
                self.inner
                    .data
                    .borrow_mut()
                    .as_mut_ptr()
                    .add(self.inner.offset) as *mut u8
            },
            shape: self.inner.shape.as_ptr(),
            stride: self.inner.stride.as_ptr(),
            dims: self.dims(),
            dtype: self.dtype as interface::DType,
        }
    }
}
