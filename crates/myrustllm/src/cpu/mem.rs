use std::rc::Rc;
use std::cell::RefCell;

use crate::common::DType;

/// If `T` implement `RawData`, `T` means that it is non-lifetime and can be copied.
pub trait RawData: Copy {
    fn dtype() -> DType;
}

impl RawData for f32 {
    fn dtype() -> DType { DType::F32 }
}
impl RawData for f64 {
    fn dtype() -> DType { DType::F64}
}
impl RawData for i32 {
    fn dtype() -> DType { DType::I32 }
}
impl RawData for i64 {
    fn dtype() -> DType { DType::I64 }
}

/// Tensor's representation in CPU memory.
#[derive(Debug)]
pub struct CPUMemory<T: RawData> {
    ptr: *mut T,
    size: usize,
    layout: std::alloc::Layout,
}

impl<T: RawData> CPUMemory<T> {
    /// Allocate a CPU memory area
    ///
    /// `size` is the number of elements
    pub fn new(size: usize) -> Self {
        let layout = std::alloc::Layout::array::<T>(size).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) as *mut T };
        CPUMemory { ptr, size, layout }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }
}

impl<T: RawData> Drop for CPUMemory<T> {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.ptr as *mut u8, self.layout);
        };
    }
}

pub type SharedCPUMemory<T> = Rc<RefCell<CPUMemory<T>>>;