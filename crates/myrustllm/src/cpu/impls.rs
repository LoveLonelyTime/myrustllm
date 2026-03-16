use std::{cell::RefCell, os::raw::c_void, rc::Rc};

use crate::{
    common::{
        DType, DTypeOf, Shape,
        dtype::{Any, DTYPE_F32, DTypeImpl, F32, F64, I32, I64},
        impls::Impl,
        init::{TensorAllocInit, TensorRawDataInit},
        io::TensorRawData,
        shape::create_contiguous_stride,
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

// Earse-Type
#[derive(Clone)]
pub struct CPUTensorAnyPrototype {
    inner: CPUTensorPrototype<u8>,
    dtype: DType,
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
            stride: create_contiguous_stride(shape),
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

macro_rules! register_tensor_prot {
    ($dt: ty, $rt: ty) => {
        impl TensorPrototype<CPU> for CPUTensorPrototype<$rt> {
            fn shape(&self) -> Shape {
                self.shape.clone()
            }

            fn dtype(&self) -> DType {
                <$dt as DTypeOf>::DTYPE
            }
            
            fn device(&self) -> <CPU as Impl>::Device {
                Default::default()
            }
        }

        impl DTypeImpl<CPU> for $dt {
            type Prototype = CPUTensorPrototype<$rt>;
        }

        impl From<CPUTensorPrototype<$rt>> for CPUTensorAnyPrototype {
            fn from(value: CPUTensorPrototype<$rt>) -> Self {
                CPUTensorAnyPrototype {
                    inner: unsafe { std::mem::transmute(value) },
                    dtype: <$dt as DTypeOf>::DTYPE,
                }
            }
        }

        impl IntoInterface for CPUTensorPrototype<$rt> {
            /// Wrap `CPUTensor<T>` into an interface struct.
            unsafe fn into_interface(&self) -> interface::CPUTensor {
                interface::CPUTensor {
                    data: unsafe {
                        self.data.borrow_mut().as_mut_ptr().add(self.offset) as *mut u8
                    },
                    shape: self.shape.as_ptr(),
                    stride: self.stride.as_ptr(),
                    dims: self.dims(),
                    dtype: <$dt as DTypeOf>::DTYPE,
                }
            }
        }
    };
}

register_tensor_prot!(F32, f32);
register_tensor_prot!(F64, f64);
register_tensor_prot!(I32, i32);
register_tensor_prot!(I64, i64);

impl TensorPrototype<CPU> for CPUTensorAnyPrototype {
    fn shape(&self) -> Shape {
        self.inner.shape.clone()
    }

    fn dtype(&self) -> DType {
        self.dtype
    }

    fn device(&self) -> <CPU as Impl>::Device {
        Default::default()
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
            dtype: self.dtype,
        }
    }
}
