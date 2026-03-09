use crate::common::{
    GenericTensor, Shape, Tensor,
    autograd::{GraphNode, OpGrad, TensorGrad},
};

// ================================================== COPY ==================================================

pub trait TensorCopy<Rhs = Self> {
    fn copy_from(&mut self, rhs: Rhs);
}

impl TensorCopy for &GenericTensor {
    fn copy_from(&mut self, rhs: Self) {
        match (self, rhs) {
            _ => todo!(),
        }
    }
}

// ================================================== COPY ==================================================

// ================================================== ADD ==================================================

impl std::ops::Add for &GenericTensor {
    type Output = GenericTensor;
    fn add(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (GenericTensor::CPUTensor(lhs), GenericTensor::CPUTensor(rhs)) => {
                GenericTensor::CPUTensor(lhs + rhs)
            }
            _ => todo!(),
        }
    }
}

#[derive(Debug)]
pub struct AddGrad {
    lhs_shape: Shape,
    rhs_shape: Shape,
}

impl AddGrad {
    pub fn new(lhs: &TensorGrad, rhs: &TensorGrad) -> Self {
        AddGrad {
            lhs_shape: lhs.shape().clone(),
            rhs_shape: rhs.shape().clone(),
        }
    }

    pub fn reduce_grad(grad: &TensorGrad, target_shape: &Shape) -> TensorGrad {
        let mut reduce_dims = Vec::new();
        let diff = grad.dims() - target_shape.len();

        for dim in 0..diff {
            reduce_dims.push(dim);
        }

        let grad = grad.add_reduce(Some(&reduce_dims), false);
        reduce_dims.clear();

        for dim in 0..target_shape.len() {
            if target_shape[dim] == 1 && grad.shape()[dim] != 1 {
                reduce_dims.push(dim);
            }
        }

        grad.add_reduce(Some(&reduce_dims), true)
    }
}

impl OpGrad for AddGrad {
    fn forward(&self, inputs: &[&GenericTensor]) -> Vec<GenericTensor> {
        assert!(
            inputs.len() == 2,
            "AddGrad requires two operands, but got {}.",
            inputs.len()
        );

        vec![inputs[0] + inputs[1]]
    }

    fn backward(&self, grad_inputs: &[&TensorGrad]) -> Vec<TensorGrad> {
        return vec![
            AddGrad::reduce_grad(grad_inputs[0], &self.lhs_shape),
            AddGrad::reduce_grad(grad_inputs[0], &self.rhs_shape),
        ];
    }
}

impl std::ops::Add for &TensorGrad {
    type Output = TensorGrad;
    fn add(self, rhs: Self) -> Self::Output {
        let op = AddGrad::new(self, rhs);
        let (node, outputs) = GraphNode::forward(op, &vec![self, rhs]);
        TensorGrad::intermediate(outputs[0].clone(), node, 0)
    }
}

// ================================================== ADD ==================================================

// ================================================== ADD_REDUCE ==================================================

pub trait TensorAddReduce {
    type Output;
    fn add_reduce(self, dims: Option<&[usize]>, keep_dim: bool) -> Self::Output;
}

impl TensorAddReduce for &GenericTensor {
    type Output = GenericTensor;
    fn add_reduce(self, dims: Option<&[usize]>, keep_dim: bool) -> Self::Output {
        match self {
            GenericTensor::CPUTensor(t) => GenericTensor::CPUTensor(t.add_reduce(dims, keep_dim)),
            _ => todo!(),
        }
    }
}

#[derive(Debug)]
pub struct AddReduceGrad {
    dims: Option<Vec<usize>>,
    keep_dim: bool,
}

impl AddReduceGrad {
    pub fn new(dims: Option<&[usize]>, keep_dim: bool) -> Self {
        AddReduceGrad {
            dims: dims.map(|r| r.iter().map(|v| *v).collect::<Vec<usize>>()),
            keep_dim,
        }
    }
}

impl OpGrad for AddReduceGrad {
    fn forward(&self, inputs: &[&GenericTensor]) -> Vec<GenericTensor> {
        vec![inputs[0].add_reduce(self.dims.as_ref().map(|v| v.as_slice()), self.keep_dim)]
    }

    fn backward(&self, grad_inputs: &[&TensorGrad]) -> Vec<TensorGrad> {
        todo!()
    }
}

impl TensorAddReduce for &TensorGrad {
    type Output = TensorGrad;
    fn add_reduce(self, dims: Option<&[usize]>, keep_dim: bool) -> Self::Output {
        let op = AddReduceGrad::new(dims, keep_dim);
        let (node, outputs) = GraphNode::forward(op, &vec![self]);

        TensorGrad::intermediate(outputs[0].clone(), node, 0)
    }
}

// ================================================== ADD_REDUCE ==================================================

// ================================================== PERMUTE ==================================================

pub trait TensorPermute {
    /// Permute the dimensions of the tensor.
    ///
    /// For `dims`, each number from `0..dims.len()` must appear just once.
    fn permute(&self, dims: &[usize]) -> Self;
}

impl TensorPermute for GenericTensor {
    fn permute(&self, dims: &[usize]) -> Self {
        match self {
            GenericTensor::CPUTensor(t) => t.permute(dims).into(),
            _ => todo!(),
        }
    }
}

pub trait TensorTranspose {
    /// Transpose the tensor.
    fn transpose(&self) -> Self;
}

impl<T: Tensor + TensorPermute> TensorTranspose for T {
    fn transpose(&self) -> Self {
        assert!(
            self.dims() > 1,
            "The dimension of the tensor to be transposed must be greater than 1, but got {}.",
            self.dims()
        );
        let dims = self.dims();
        let mut permute_dims: Vec<usize> = (0..dims - 2).collect();
        // Permute the last two dimensions
        permute_dims.extend([dims - 1, dims - 2]);

        self.permute(&permute_dims)
    }
}

#[derive(Debug)]
pub struct PermuteGrad {
    dims: Vec<usize>,
}

impl PermuteGrad {
    pub fn new(dims: &[usize]) -> Self {
        PermuteGrad {
            dims: Vec::from(dims),
        }
    }
}

impl OpGrad for PermuteGrad {
    fn forward(&self, inputs: &[&GenericTensor]) -> Vec<GenericTensor> {
        assert!(inputs.len() == 1);
        vec![inputs[0].permute(&self.dims)]
    }

    fn backward(&self, grad_inputs: &[&TensorGrad]) -> Vec<TensorGrad> {
        let mut inv = vec![0; self.dims.len()];
        for i in 0..self.dims.len() {
            inv[self.dims[i]] = i;
        }

        vec![grad_inputs[0].permute(&self.dims)]
    }
}

impl TensorPermute for TensorGrad {
    fn permute(&self, dims: &[usize]) -> Self {
        let op = PermuteGrad::new(dims);
        let (node, outputs) = GraphNode::forward(op, &vec![self]);
        TensorGrad::intermediate(outputs[0].clone(), node, 0)
    }
}

// ================================================== PERMUTE ==================================================

// ================================================== MATMUL ==================================================

pub trait TensorMatmul<Rhs = Self> {
    type Output;
    fn matmul(self, rhs: Rhs) -> Self::Output;
}

impl TensorMatmul for &GenericTensor {
    type Output = GenericTensor;
    fn matmul(self, rhs: Self) -> Self::Output {
        match (self, rhs) {
            (GenericTensor::CPUTensor(lhs), GenericTensor::CPUTensor(rhs)) => {
                GenericTensor::CPUTensor(lhs.matmul(rhs))
            }
            _ => todo!(),
        }
    }
}

#[derive(Debug)]
pub struct MatmulGrad {
    lhs: TensorGrad,
    rhs: TensorGrad,
}

impl MatmulGrad {
    pub fn new(lhs: &TensorGrad, rhs: &TensorGrad) -> Self {
        MatmulGrad {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        }
    }
}

impl OpGrad for MatmulGrad {
    fn forward(&self, inputs: &[&GenericTensor]) -> Vec<GenericTensor> {
        assert!(inputs.len() == 2);
        vec![inputs[0].matmul(inputs[1])]
    }

    fn backward(&self, grad_inputs: &[&TensorGrad]) -> Vec<TensorGrad> {
        vec![
            grad_inputs[0].matmul(&self.rhs.transpose()),
            self.lhs.transpose().matmul(grad_inputs[0]),
        ]
    }
}

impl TensorMatmul for &TensorGrad {
    type Output = TensorGrad;

    fn matmul(self, rhs: Self) -> Self::Output {
        let op = MatmulGrad::new(self, rhs);
        let (node, outputs) = GraphNode::forward(op, &[self, rhs]);
        TensorGrad::intermediate(outputs[0].clone(), node, 0)
    }
}

// ================================================== MATMUL ==================================================
