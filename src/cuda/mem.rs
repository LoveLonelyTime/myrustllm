use crate::cuda::interface;
use std::marker::PhantomData;

/// Declare CUDA types
///
/// All valid CUDA types must be implement this trait
pub trait CUDAType {
    fn size() -> usize;
}

pub struct CUDAI32;
pub struct CUDAF16;
pub struct CUDAF32;
pub struct CUDAF64;


impl CUDAType for CUDAF16 {
    fn size() -> usize {
        2
    }
}
impl CUDAType for CUDAF32 {
    fn size() -> usize {
        4
    }
}
impl CUDAType for CUDAF64 {
    fn size() -> usize {
        8
    }
}

/// Wrapper of CUDA memory
pub struct CUDAMemory<T: CUDAType> {
    ptr: *mut libc::c_void,
    size: usize,
    _marker: PhantomData<T>,
}

impl<T: CUDAType> CUDAMemory<T> {
    /// Alloc a CUDA memory area
    ///
    /// `size` is the number of elements
    pub fn new(size: usize) -> Self {
        let ptr = unsafe { interface::cuda_tensor_alloc(size * T::size()) };

        CUDAMemory {
            ptr,
            size,
            _marker: PhantomData,
        }
    }

    /// Return a constant pointer of the memory area
    ///
    /// **Warning: This pointer ONLY can be used in CUDA**
    pub fn as_ptr(&self) -> *const libc::c_void {
        self.ptr
    }

    /// Return a mutable pointer of the memory area
    ///
    /// **Warning: This pointer ONLY can be used in CUDA**
    pub fn as_mut_ptr(&mut self) -> *mut libc::c_void {
        self.ptr
    }

    /// Return the number of elements
    pub fn size(&self) -> usize {
        self.size
    }
}

impl<T: CUDAType> Drop for CUDAMemory<T> {
    fn drop(&mut self) {
        unsafe {
            interface::cuda_tensor_destroy(self.ptr);
        };
    }
}
