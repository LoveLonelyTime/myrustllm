use crate::cpu::tensor::Tensor;
use crate::cuda::interface;
use crate::cuda::mem::CUDAF32;
use crate::cuda::tensor::{CUDATensor, broadcast};

impl std::ops::Add for &CUDATensor<CUDAF32> {
    type Output = CUDATensor<CUDAF32>;
    fn add(self, rhs: Self) -> Self::Output {
        let (a, b) = broadcast(self, rhs).expect(&format!(
            "Two tensors with shapes ({}, {}) cannot be broadcast.",
            self.shape(),
            rhs.shape()
        ));

        let dims = a.dims();
        let mut out = CUDATensor::from_shape(&a.shape());

        unsafe {
            let a_ptr = a.borrow().as_ptr() as *const libc::c_float;
            let b_ptr = b.borrow().as_ptr() as *const libc::c_float;
            let c_ptr = out.borrow_mut().as_mut_ptr() as *mut libc::c_float;
            interface::cuda_tensor_add_f32(
                a_ptr.add(a.offset()),
                b_ptr.add(b.offset()),
                c_ptr.add(out.offset()),
                a.stride().as_ptr(),
                b.stride().as_ptr(),
                out.stride().as_ptr(),
                a.shape().as_ptr(),
                dims,
                a.shape().numel(),
            );
        };

        out
    }
}

// impl std::ops::Neg for &CUDATensor<CUDAF32> {
//     type Output = CUDATensor<CUDAF32>;

//     fn neg(self) -> Self::Output {

//     }
// }
