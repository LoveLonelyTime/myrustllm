use crate::common::dtype::Scalar;
use crate::common::{DType, Shape, Tensor};
use crate::cpu::dynamic::CPUGenericTensor;
use crate::cpu::interface;
use crate::cpu::mem::{CPUMemory, RawData};
use crate::cpu::tensor::CPUTensor;
use num_traits::{One, Zero};

// ================================================== ALLOC ==================================================

impl<T: RawData> CPUTensor<T> {
    pub fn alloc(shape: &Shape) -> Self {
        CPUTensor::new(
            CPUMemory::new(shape.numel()).into(),
            shape,
            &Shape::create_contiguous_stride(shape),
            0,
        )
    }

    pub fn alloc_like<U: RawData>(tensor: CPUTensor<U>) -> Self {
        CPUTensor::alloc(&tensor.shape())
    }
}

impl CPUGenericTensor {
    pub fn alloc(shape: &Shape, dtype: DType) -> Self {
        match dtype {
            DType::F32 => CPUTensor::<f32>::alloc(shape).into(),
            _ => todo!(),
        }
    }

    pub fn alloc_like(tensor: &CPUGenericTensor) -> Self {
        CPUGenericTensor::alloc(&tensor.shape(), tensor.dtype())
    }
}

// ================================================== ALLOC ==================================================

// ================================================== LITERAL ==================================================

/// Literal
pub trait Literal<T> {
    /// Return the original shape
    fn shape(&self) -> Shape;

    /// Flatten the literal and append to `out`
    fn flatten(&self, out: &mut Vec<T>);
}

// The recursive implementation for array
impl<T, U: Literal<T>, const N: usize> Literal<T> for [U; N] {
    fn shape(&self) -> Shape {
        let mut shape_v = vec![N];
        if N > 0 {
            shape_v.extend(self[0].shape().iter());
        }
        shape_v.into()
    }

    fn flatten(&self, out: &mut Vec<T>) {
        for item in self {
            item.flatten(out);
        }
    }
}

// The implementation for raw data
impl<T: RawData> Literal<T> for T {
    fn shape(&self) -> Shape {
        Shape::scalar()
    }

    fn flatten(&self, out: &mut Vec<T>) {
        out.push(*self);
    }
}

impl<T: RawData> CPUTensor<T> {
    /// Create a new CPU tensor from the array `input`.
    ///
    /// # Examples
    ///
    /// ```
    /// use myrustllm::cpu::tensor::CPUTensor;
    ///
    /// let tensor = CPUTensor::<f32>::literal([
    ///     [1.0, 1.0],
    ///     [2.0, 2.0]
    /// ]);
    /// ```
    pub fn literal<U: Literal<T>>(input: U) -> Self {
        let mut data = Vec::new();
        input.flatten(&mut data);

        let shape = input.shape();
        let stride = Shape::create_contiguous_stride(&shape);
        let mut mem = CPUMemory::new(shape.numel());

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), mem.as_mut_ptr(), data.len());
        };

        CPUTensor::new(mem.into(), &shape, &stride, 0)
    }
}

// ================================================== LITERAL ==================================================

// ================================================== FILL ==================================================

pub trait TensorFillInitBase: RawData {
    fn fill(tensor: &mut CPUTensor<Self>, val: Self);
}

impl TensorFillInitBase for f32 {
    fn fill(tensor: &mut CPUTensor<Self>, val: Self) {
        unsafe {
            interface::cpu_tensor_init_fill_f32_(tensor.into_interface(), val);
        };
    }
}

impl<T: TensorFillInitBase> CPUTensor<T> {
    pub fn fill(shape: &Shape, val: T) -> Self {
        let mut tensor = CPUTensor::alloc(shape);
        T::fill(&mut tensor, val);
        tensor
    }

    pub fn fill_(&mut self, val: T) {
        T::fill(self, val);
    }

    pub fn fill_like<U: RawData>(tensor: &CPUTensor<U>, val: T) -> Self {
        CPUTensor::fill(&tensor.shape(), val)
    }

    pub fn scalar(val: T) -> Self {
        CPUTensor::fill(&Shape::scalar(), val)
    }
}

impl<T: TensorFillInitBase + Zero> CPUTensor<T> {
    pub fn zeros(shape: &Shape) -> Self {
        CPUTensor::fill(shape, T::zero())
    }

    pub fn zeros_(&mut self) {
        CPUTensor::fill_(self, T::zero())
    }

    pub fn zeros_like<U: RawData>(tensor: &CPUTensor<U>) -> Self {
        CPUTensor::zeros(&tensor.shape())
    }
}

impl<T: TensorFillInitBase + One> CPUTensor<T> {
    pub fn ones(shape: &Shape) -> Self {
        CPUTensor::fill(shape, T::one())
    }

    pub fn ones_(&mut self) {
        CPUTensor::fill_(self, T::one())
    }

    pub fn ones_like<U: RawData>(tensor: &CPUTensor<U>) -> Self {
        CPUTensor::ones(&tensor.shape())
    }
}

impl CPUGenericTensor {
    pub fn fill(shape: &Shape, val: Scalar) -> Self {
        match val {
            Scalar::F32(val) => CPUTensor::<f32>::fill(shape, val).into(),
            _ => todo!(),
        }
    }

    pub fn fill_(&mut self, val: Scalar) {
        match (self, val) {
            (CPUGenericTensor::F32(t), Scalar::F32(val)) => t.fill_(val),
            _ => todo!(),
        }
    }

    pub fn fill_like(tensor: &CPUGenericTensor, val: Scalar) -> Self {
        CPUGenericTensor::fill(&tensor.shape(), val)
    }

    pub fn scalar(val: Scalar) -> Self {
        CPUGenericTensor::fill(&Shape::scalar(), val)
    }

    pub fn zeros(shape: &Shape, dtype: DType) -> Self {
        match dtype {
            DType::F32 => CPUTensor::<f32>::zeros(shape).into(),
            _ => todo!(),
        }
    }

    pub fn zeros_(&mut self) {
        match self {
            CPUGenericTensor::F32(t) => t.zeros_(),
            _ => todo!(),
        }
    }

    pub fn zeros_like(tensor: &CPUGenericTensor) -> Self {
        CPUGenericTensor::zeros(&tensor.shape(), tensor.dtype())
    }

    pub fn ones(shape: &Shape, dtype: DType) -> Self {
        match dtype {
            DType::F32 => CPUTensor::<f32>::ones(shape).into(),
            _ => todo!(),
        }
    }

    pub fn ones_(&mut self) {
        match self {
            CPUGenericTensor::F32(t) => t.ones_(),
            _ => todo!(),
        }
    }

    pub fn ones_like(tensor: &CPUGenericTensor) -> Self {
        CPUGenericTensor::ones(&tensor.shape(), tensor.dtype())
    }
}

// ================================================== FILL ==================================================

// ================================================== NORMAL ==================================================

pub trait TensorNormalInitBase: RawData {
    fn uniform(tensor: &mut CPUTensor<Self>, mean: f32, std: f32);
}

impl TensorNormalInitBase for f32 {
    fn uniform(tensor: &mut CPUTensor<Self>, mean: f32, std: f32) {
        unsafe {
            interface::cpu_tensor_init_normal_f32_(tensor.into_interface(), mean, std, 0);
        };
    }
}

impl<T: TensorNormalInitBase> CPUTensor<T> {
    pub fn uniform(shape: &Shape, mean: f32, std: f32) -> Self {
        let mut tensor = CPUTensor::alloc(shape);
        T::uniform(&mut tensor, mean, std);
        tensor
    }

    pub fn uniform_(&mut self, mean: f32, std: f32) {
        T::uniform(self, mean, std);
    }

    pub fn uniform_like<U: RawData>(tensor: &CPUTensor<U>, mean: f32, std: f32) -> Self {
        CPUTensor::uniform(&tensor.shape(), mean, std)
    }
}

impl CPUGenericTensor {
    pub fn uniform(shape: &Shape, mean: f32, std: f32, dtype: DType) -> Self {
        match dtype {
            DType::F32 => CPUTensor::<f32>::uniform(shape, mean, std).into(),
            _ => todo!(),
        }
    }

    pub fn uniform_(&mut self, mean: f32, std: f32) {
        match self {
            CPUGenericTensor::F32(t) => t.uniform_(mean, std),
            _ => todo!(),
        }
    }

    pub fn uniform_like(tensor: &CPUGenericTensor, mean: f32, std: f32) -> Self {
        CPUGenericTensor::uniform(&tensor.shape(), mean, std, tensor.dtype())
    }
}

// ================================================== NORMAL ==================================================
