/// Wrapper for CPU memory
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

    /// Return a constant pointer of the memory area
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Return a mutable pointer of the memory area
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr
    }

    /// Return the number of elements
    pub fn size(&self) -> usize {
        self.size
    }
}

impl<T> Drop for CPUMemory<T> {
    fn drop(&mut self) {
        unsafe {
            std::alloc::dealloc(self.ptr as *mut u8, self.layout);
        };
    }
}
