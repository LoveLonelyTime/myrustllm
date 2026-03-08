use crate::cuda::{
    mem::{CUDAF32, CUDAType},
    tensor::CUDATensor,
};

// pub fn cuda_gemm(a: &CUDATensor<CUDAF32>, b: &CUDATensor<CUDAF32>) -> CUDATensor<CUDAF32> {
    
//     unsafe {
//         let a_ptr = a.borrow().as_ptr() as *const libc::c_float;
//         let b_ptr = b.borrow().as_ptr() as *const libc::c_float;
//         let out_ptr = out.borrow_mut().as_mut_ptr() as *mut libc::c_float;
//         interface::cuda_tensor_add_f32(
//             a_ptr.add(a.offset()),
//             b_ptr.add(b.offset()),
//             out_ptr.add(out.offset()),
//             a.stride().as_ptr(),
//             b.stride().as_ptr(),
//             out.stride().as_ptr(),
//             a.shape().as_ptr(),
//             dims,
//             a.shape().numel(),
//         );
//     };
// }
