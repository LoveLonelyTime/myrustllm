/// Device enum.
///
/// e.g.: CPU, CUDA(0), CUDA(1)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Device {
    CPU,
    CUDA(usize),
}

// pub trait Device {}

// pub struct CPU {}

// impl Device for CPU {}
