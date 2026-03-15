use std::cell::RefCell;
use std::rc::Rc;

/// Tensor's representation in CPU memory.
#[derive(Debug)]
pub struct CPUMemory<T> {
    ptr: *mut T,
    size: usize,
    layout: std::alloc::Layout,
}

impl<T> CPUMemory<T> {
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

impl<T> Drop for CPUMemory<T> {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.ptr as *mut u8, self.layout);
        };
    }
}

impl<T> From<CPUMemory<T>> for SharedCPUMemory<T> {
    fn from(value: CPUMemory<T>) -> Self {
        Rc::new(RefCell::new(value))
    }
}

pub type SharedCPUMemory<T> = Rc<RefCell<CPUMemory<T>>>;

