/// DType enum.
///
/// e.g.: F32, I32, ...
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType{
    F32,
    F64,
    I32,
    I64,
}
pub enum Scalar {
    F32(f32),
    F64(f64),
    I32(i32),
    I64(i64),
}
