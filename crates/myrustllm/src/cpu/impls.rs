use crate::common::dtype::{F32, F64, I32, I64};
use crate::common::shape::create_contiguous_stride;
use crate::common::tensor::TensorPrototype;
use crate::common::{DType, DTypeImpl, DTypeOf, Impl, Shape};
use crate::cpu::interface;
use crate::cpu::interface::IntoInterface;
use crate::cpu::mem::{CPUMemory, SharedCPUMemory};
use std::env;

/// CPU is one of the basic implementations in MyRustLLM, which provides tensor operations on CPU.
/// All Tensors using CPU implementation will be allocated in the main memory and processed by the CPU.
/// Tensors on CPU use shared memory to store their data, so it can be easily cloned with few costs.

#[derive(Debug)]
pub struct CPU {}

impl Impl for CPU {
    /// CPU does not require any special device context, so we can use an empty tuple as the device type.
    type Device = ();
}

/// `CPUTensorPrototype` is the prototype of a tensor on CPU. It contains shared memory for the tensor data, as well as the shape, stride, and offset information.
#[derive(Debug)]
pub struct CPUTensorPrototype<T> {
    data: SharedCPUMemory<T>,
    shape: Shape,
    stride: Shape,
    offset: usize,
}

impl<T> Clone for CPUTensorPrototype<T> {
    fn clone(&self) -> Self {
        CPUTensorPrototype {
            data: self.data.clone(),
            shape: self.shape.clone(),
            stride: self.stride.clone(),
            offset: self.offset,
        }
    }
}

impl<T> CPUTensorPrototype<T> {
    /// Create a new CPU tensor prototype with the given shared data, shape, stride, and offset.
    pub fn new(data: SharedCPUMemory<T>, shape: &Shape, stride: &Shape, offset: usize) -> Self {
        CPUTensorPrototype {
            data,
            shape: shape.clone(),
            stride: stride.clone(),
            offset,
        }
    }

    /// Allocate a new CPU tensor prototype with the given shape. The data won't be initialized, and the caller should fill it with valid data before using it.
    pub fn alloc(shape: &Shape) -> Self {
        CPUTensorPrototype {
            data: CPUMemory::new(shape.numel()).into(),
            shape: shape.clone(),
            stride: create_contiguous_stride(shape),
            offset: 0,
        }
    }

    /// Return the stride of the tensor.
    pub fn stride(&self) -> Shape {
        self.stride.clone()
    }

    /// Return the offset of the tensor.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Return a new reference to the shared memory of the tensor data.
    pub fn data(&self) -> SharedCPUMemory<T> {
        self.data.clone()
    }
}

macro_rules! register_tensor_prot {
    // $dt: data type, $rt: Rust type
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
    };
}

register_tensor_prot!(F32, f32);
register_tensor_prot!(F64, f64);
register_tensor_prot!(I32, i32);
register_tensor_prot!(I64, i64);

impl<T> IntoInterface for CPUTensorPrototype<T>
where
    CPUTensorPrototype<T>: TensorPrototype<CPU>,
{
    // Wrap `CPUTensor<T>` into an interface struct.
    unsafe fn into_interface(&self) -> interface::CPUTensor {
        interface::CPUTensor {
            data: unsafe { self.data.borrow_mut().as_mut_ptr().add(self.offset) as *mut u8 },
            shape: self.shape.as_ptr(),
            stride: self.stride.as_ptr(),
            dims: self.dims(),
            dtype: self.dtype(),
        }
    }
}

// This can let users use `println!` for CPUTensor
impl<T: std::fmt::Display> std::fmt::Display for CPUTensorPrototype<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let limit = env::var("MYRUSTLLM_DISPLAY_LIMIT")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(3);

        let mut trace = vec![0; self.shape.len()];

        fn _fmt_recursive<T: std::fmt::Display>(
            limit: usize,
            f: &mut std::fmt::Formatter<'_>,
            tensor: &CPUTensorPrototype<T>,
            trace: &mut [usize],
            dim: usize,
        ) -> std::fmt::Result {
            if dim == tensor.shape.len() {
                let loc = trace
                    .iter()
                    .zip(tensor.stride.iter())
                    .fold(tensor.offset, |offset, (&i, &s)| offset + i * s);
                return write!(f, "{}", tensor.data.borrow()[loc]);
            }

            write!(f, "[")?;
            if tensor.shape[dim] <= limit * 2 {
                for i in 0..tensor.shape[dim] {
                    trace[dim] = i;
                    _fmt_recursive(limit, f, tensor, trace, dim + 1)?;
                    write!(f, ", ")?;
                }
            } else {
                // Front
                for i in 0..limit {
                    trace[dim] = i;
                    _fmt_recursive(limit, f, tensor, trace, dim + 1)?;
                    write!(f, ", ")?;
                }

                write!(f, ", ..., ")?;

                // Back
                for i in tensor.shape[dim] - limit..tensor.shape[dim] {
                    trace[dim] = i;
                    _fmt_recursive(limit, f, tensor, trace, dim + 1)?;
                    write!(f, ", ")?;
                }
            }
            write!(f, "]")?;
            Ok(())
        }

        _fmt_recursive(limit, f, self, trace.as_mut_slice(), 0)
    }
}
