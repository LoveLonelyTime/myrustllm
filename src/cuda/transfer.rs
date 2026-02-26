use crate::cpu::mem::CPUMemory;
use crate::cpu::tensor::CPUTensor;
use crate::cpu::tensor::Tensor;
use crate::cuda::interface;
use crate::cuda::mem::{CUDAF32, CUDAF64, CUDAMemory, CUDAType};
use crate::cuda::tensor::CUDATensor;

// ---------- CUDAF16 ----------

// ---------- CUDAF16 END ----------

// ---------- CUDAF32 ----------

impl From<&CPUTensor<f32>> for CUDATensor<CUDAF32> {
    fn from(value: &CPUTensor<f32>) -> Self {
        let size = value.get_memory_stride();
        let mut mem = CUDAMemory::<CUDAF32>::new(size);
        unsafe {
            interface::cuda_tensor_copy_from_host(
                mem.as_mut_ptr(),
                value.as_ptr() as *const libc::c_void,
                size * CUDAF32::size(),
            );
        };

        CUDATensor::new(mem, &value.shape(), &value.stride(), 0)
    }
}

impl From<&CUDATensor<CUDAF32>> for CPUTensor<f32> {
    fn from(value: &CUDATensor<CUDAF32>) -> Self {
        let size = value.get_memory_stride();
        let mut mem = CPUMemory::<f32>::new(size);

        unsafe {
            interface::cuda_tensor_copy_from_device(
                mem.as_mut_ptr() as *mut libc::c_void,
                value
                    .borrow()
                    .as_ptr()
                    .byte_add(value.offset() * CUDAF32::size()),
                size * CUDAF32::size(),
            );
        };

        CPUTensor::new(mem, &value.shape(), &value.stride(), 0)
    }
}

// ---------- CUDAF32 END ----------

// ---------- CUDAF64 ----------

impl From<&CPUTensor<f64>> for CUDATensor<CUDAF64> {
    fn from(value: &CPUTensor<f64>) -> Self {
        let size = value.get_memory_stride();
        let mut mem = CUDAMemory::<CUDAF64>::new(size);
        unsafe {
            interface::cuda_tensor_copy_from_host(
                mem.as_mut_ptr(),
                value.as_ptr() as *const libc::c_void,
                size * CUDAF64::size(),
            );
        };

        CUDATensor::new(mem, &value.shape(), &value.stride(), 0)
    }
}

impl From<&CUDATensor<CUDAF64>> for CPUTensor<f64> {
    fn from(value: &CUDATensor<CUDAF64>) -> Self {
        let size = value.get_memory_stride();
        let mut mem = CPUMemory::<f64>::new(size);

        unsafe {
            interface::cuda_tensor_copy_from_device(
                mem.as_mut_ptr() as *mut libc::c_void,
                value
                    .borrow()
                    .as_ptr()
                    .byte_add(value.offset() * CUDAF64::size()),
                size * CUDAF64::size(),
            );
        };

        CPUTensor::new(mem, &value.shape(), &value.stride(), 0)
    }
}

// ---------- CUDAF64 END ----------
